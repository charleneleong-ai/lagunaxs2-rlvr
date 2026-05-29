"""Multi-turn tool-use env over general-agent-style tasks (Pydantic DB + tool APIs + verifier).

Each task gives the agent an NL instruction and a set of tools; the agent emits tool calls (a
```python block) each turn, the env replays the accumulated calls in a subprocess and returns
captured stdout as the observation, and scores with the task's verifier. This exercises Laguna's
interleaved tool-calling. Reward = task solved + an efficiency nudge for fewer turns.

Tasks come from a JSONL corpus (the synthesizer's output) or a small builtin tiered set. State is
kept as the list of tool-call strings and replayed statelessly each turn — no live objects in env
state, and the same `run_solution` used to validate tasks scores them. No sandbox: $0 with any model.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import verifiers as vf
from datasets import Dataset

# --- vendored from src/laguna_rlvr: this env ships as a standalone Hub wheel and laguna_rlvr
#     isn't a PyPI dep, so it can't be a requirement. Trimmed fork of just the subset used
#     (code_exec + rewards + synth.task) — not a byte-for-byte mirror of the source. ---
_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block; fall back to the whole text."""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


@dataclass(frozen=True, slots=True)
class RolloutState:
    tests_passed: int
    tests_total: int
    turns: int
    max_turns: int
    succeeded: bool


def binary(s: RolloutState) -> float:
    return float(s.succeeded)


def shaped(s: RolloutState, efficiency_weight: float = 0.1) -> float:
    """Dense reward: test-pass fraction plus an efficiency nudge paid only on success."""
    partial = s.tests_passed / s.tests_total if s.tests_total > 0 else 0.0
    bonus = max(0.0, 1.0 - s.turns / s.max_turns) if s.succeeded and s.max_turns > 0 else 0.0
    return partial + efficiency_weight * bonus


_HARNESS = "import sys as _s; _s.exit(0 if verify() else 1)"


@dataclass(frozen=True, slots=True)
class Task:
    domain: str
    tier: int
    schema_code: str
    tools_code: str
    instruction: str
    gold_solution: str
    verifier_code: str

    def to_dict(self) -> dict:
        return asdict(self)


def _assemble_program(task: Task, solution: str | None = None) -> str:
    """schema + tools + (a candidate solution, default the gold) + verifier + exit-on-verify."""
    return "\n\n".join([task.schema_code, task.tools_code,
                        task.gold_solution if solution is None else solution,
                        task.verifier_code, _HARNESS])


def run_solution(task: Task, solution: str | None = None, timeout: float = 5.0) -> bool:
    """True iff the solution drives the task's verifier to pass (subprocess-isolated)."""
    try:
        result = subprocess.run([sys.executable, "-c", _assemble_program(task, solution)],
                                capture_output=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False

_TASK_FIELDS = ("domain", "tier", "schema_code", "tools_code", "instruction", "gold_solution", "verifier_code")

# --- builtin tiered tasks (day_spa domain) — each is self-consistent (gold satisfies verifier) ---
_SCHEMA = ("from pydantic import BaseModel\n"
           "class Service(BaseModel):\n    name: str\n    price: float\n"
           "class Appointment(BaseModel):\n    service: str")
_TOOLS = (
    "db = {'appointments': [], 'services': [Service(name='swedish', price=80.0),"
    " Service(name='deep tissue', price=120.0), Service(name='facial', price=60.0)]}\n"
    "def list_services():\n    return [(s.name, s.price) for s in db['services']]\n"
    "def book_appointment(service):\n    db['appointments'].append(Appointment(service=service)); return 'booked'"
)
_VERIFY_DEEP = "def verify():\n    return any(a.service == 'deep tissue' for a in db['appointments'])"
_VERIFY_CHEAP = ("def verify():\n    cheapest = min(db['services'], key=lambda s: s.price).name\n"
                 "    return any(a.service == cheapest for a in db['appointments'])")

_BUILTIN_TASKS = [
    Task("day_spa", 0, _SCHEMA, _TOOLS, "Book a 'deep tissue' appointment.",
         'book_appointment("deep tissue")', _VERIFY_DEEP),
    Task("day_spa", 1, _SCHEMA, _TOOLS,
         "Book the single cheapest service. Use list_services() to inspect prices first.",
         "_c = min(db['services'], key=lambda s: s.price).name\nbook_appointment(_c)", _VERIFY_CHEAP),
]


def _tool_signatures(tools_code: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"^def (\w+\([^)]*\))", tools_code, re.M)
            if not m.group(1).startswith("_")]


def _load_tasks(source: str) -> list[Task]:
    if source == "builtin":
        return _BUILTIN_TASKS
    rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    return [Task(**{k: r[k] for k in _TASK_FIELDS}) for r in rows]


def _text(message) -> str:
    if isinstance(message, str):
        return message
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") for p in content)
    return ""


def _prompt(task: Task) -> str:
    tools = "\n".join(f"  - {sig}" for sig in _tool_signatures(task.tools_code))
    return (f"You are an agent that completes a task by calling tools.\n\nTask: {task.instruction}\n\n"
            f"Available tools:\n{tools}\n\n"
            "Each turn, reply with a ```python code block of tool calls. print(...) any results you "
            "want to observe. You'll get the stdout back. Continue until the task is done.")


class GeneralAgentEnv(vf.MultiTurnEnv):
    def __init__(self, tasks: list[Task], *, max_turns: int, timeout: float,
                 efficiency_weight: float, **kwargs):
        self._timeout = timeout
        rows = [{"question": _prompt(t), "answer": "", "info": t.to_dict()} for t in tasks]
        ds = Dataset.from_list(rows)
        rubric = vf.Rubric(funcs=[self._reward_fn(max_turns, efficiency_weight), self._success_fn(max_turns)],
                           weights=[1.0, 0.0])
        # dataset → RL training (prime train reads get_dataset()); eval_dataset → probe (prime eval run)
        super().__init__(dataset=ds, eval_dataset=ds, rubric=rubric,
                         max_turns=max_turns, message_type="chat", **kwargs)

    @staticmethod
    def _task(state) -> Task:
        info = state["info"]
        return Task(**{k: info[k] for k in _TASK_FIELDS})

    def _rs(self, state, max_turns: int) -> RolloutState:
        return RolloutState(tests_passed=int(state.get("solved", False)), tests_total=1,
                            turns=int(state.get("turn", 0)), max_turns=max_turns,
                            succeeded=bool(state.get("solved", False)))

    def _reward_fn(self, max_turns: int, efficiency_weight: float):
        def reward(state, **_) -> float:
            return shaped(self._rs(state, max_turns), efficiency_weight)
        return reward

    def _success_fn(self, max_turns: int):
        def success(state, **_) -> float:
            return binary(self._rs(state, max_turns))
        return success

    async def setup_state(self, state) -> None:
        state["solved"] = False
        state["calls"] = []

    @vf.stop
    async def is_solved(self, state) -> bool:
        return bool(state.get("solved", False))

    def _observe(self, task: Task, calls: list[str]) -> str:
        program = f"{task.schema_code}\n\n{task.tools_code}\n\n" + "\n".join(calls)
        try:
            r = subprocess.run([sys.executable, "-c", program], capture_output=True,
                               text=True, timeout=self._timeout)
            return (r.stdout + r.stderr)[-2000:] if (r.stdout or r.stderr) else "(no output)"
        except subprocess.TimeoutExpired:
            return "(timed out)"

    async def env_response(self, messages, state, **kwargs):
        task = self._task(state)
        state["calls"].append(extract_code(_text(messages[-1])))
        observation = self._observe(task, state["calls"])
        if run_solution(task, "\n".join(state["calls"]), self._timeout):
            state["solved"] = True
            return [{"role": "user", "content": f"{observation}\n\n✅ Task complete."}]
        return [{"role": "user", "content": f"{observation}\n\nNot complete yet — continue with more tool calls."}]


def load_environment(source: str = "builtin", max_turns: int = 6, timeout: float = 5.0,
                     efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    return GeneralAgentEnv(_load_tasks(source), max_turns=max_turns, timeout=timeout,
                           efficiency_weight=efficiency_weight)
