"""Multi-turn tool-use env over general-agent-style tasks (Pydantic DB + tool APIs + verifier).

Each task gives the agent an NL instruction and a set of tools; the agent emits tool calls (a
```python block) each turn. The env replays the accumulated calls in ONE subprocess (via
`task.run_capturing`) that returns both the captured stdout (the observation) and the verifier's
verdict — exercising Laguna's interleaved tool-calling. Reward = task solved + an efficiency nudge.

Tasks come from a JSONL corpus (the synthesizer's output) or a small builtin tiered set. State is
just the list of tool-call strings, replayed statelessly each turn — no live objects in env state,
and the same `run_capturing` that validates tasks scores them. No sandbox: $0 with any model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from laguna_rlvr.code_exec import extract_code, message_text   # vendor before any Hub push
from laguna_rlvr.rewards import RolloutState, binary, shaped
from laguna_rlvr.synth.task import Task, run_capturing

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
    return [Task(**json.loads(line)) for line in Path(source).read_text().splitlines() if line.strip()]


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
        self._eff_w = efficiency_weight
        rows = [{"question": _prompt(t), "answer": "", "info": t.to_dict()} for t in tasks]
        super().__init__(eval_dataset=Dataset.from_list(rows),
                         rubric=vf.Rubric(funcs=[self._reward, self._success], weights=[1.0, 0.0]),
                         max_turns=max_turns, message_type="chat", **kwargs)

    def _rs(self, state) -> RolloutState:
        solved = bool(state.get("solved", False))
        return RolloutState(tests_passed=int(solved), tests_total=1, turns=int(state.get("turn", 0)),
                            max_turns=self.max_turns, succeeded=solved)

    def _reward(self, state, **_) -> float:
        return shaped(self._rs(state), self._eff_w)

    def _success(self, state, **_) -> float:
        return binary(self._rs(state))

    async def setup_state(self, state) -> None:
        state["solved"] = False
        state["calls"] = []

    @vf.stop
    async def is_solved(self, state) -> bool:
        return bool(state.get("solved", False))

    async def env_response(self, messages, state, **kwargs):
        state["calls"].append(extract_code(message_text(messages[-1])))
        solved, observation = run_capturing(Task(**state["info"]), "\n".join(state["calls"]), self._timeout)
        state["solved"] = solved
        suffix = "\n\n✅ Task complete." if solved else "\n\nNot complete yet — continue with more tool calls."
        return [{"role": "user", "content": (observation[-2000:] or "(no output)") + suffix}]


def load_environment(source: str = "builtin", max_turns: int = 6, timeout: float = 5.0,
                     efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    return GeneralAgentEnv(_load_tasks(source), max_turns=max_turns, timeout=timeout,
                           efficiency_weight=efficiency_weight)
