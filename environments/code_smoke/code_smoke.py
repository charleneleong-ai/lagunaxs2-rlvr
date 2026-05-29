"""Multi-turn, no-Docker code-repair env: write a function, run its tests, fix on failures, repeat.

Tasks come from MBPP (a real difficulty spread, so a strong model's pass rate is intermediate ->
a learnable RL gradient), with a small `builtin` set for offline/tests. Reward is dense shaped
partial credit (test-pass fraction + an efficiency nudge for solving in fewer turns).

verifiers 0.1.14 MultiTurnEnv API: override setup_state (mutate in place) and env_response
(-> Messages); completion via the @vf.stop method; scoring via the rubric. No sandbox — tests
run in a local subprocess (code_exec.score_code), so it's $0 with any OpenAI-compatible model.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

import verifiers as vf
from datasets import Dataset, load_dataset

# --- vendored from src/laguna_rlvr: the env ships as a standalone Hub wheel and laguna_rlvr
#     isn't a PyPI dep, so it can't be a requirement. This is a trimmed fork of just the subset
#     code_smoke needs (code_exec + rewards) — not a byte-for-byte mirror of the source. ---
_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block; fall back to the whole text."""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def score_code(code: str, tests: list[str], timeout: float = 5.0) -> tuple[int, int]:
    """Return (passed, total): how many assert-tests the code satisfies, each in its own subprocess."""
    passed = 0
    for test in tests:
        try:
            result = subprocess.run([sys.executable, "-c", f"{code}\n{test}\n"],
                                    capture_output=True, timeout=timeout)
            passed += result.returncode == 0
        except subprocess.TimeoutExpired:
            pass
    return passed, len(tests)


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


_REPLY = "\nReply with a ```python code block."

# (prompt, tests, setup) — setup is scaffolding (imports) prepended before running the asserts.
_BUILTIN_TASKS = [
    ("Write a Python function `add(a, b)` that returns their sum." + _REPLY,
     ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"], ""),
    ("Write a Python function `is_palindrome(s)` returning True iff s reads the same backwards." + _REPLY,
     ["assert is_palindrome('racecar')", "assert not is_palindrome('hello')", "assert is_palindrome('')"], ""),
    ("Write a Python function `fib(n)` returning the n-th Fibonacci number (fib(0)=0, fib(1)=1)." + _REPLY,
     ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(10) == 55"], ""),
]


def _load_tasks(source: str, n_tasks: int | None, start: int,
                prompt_field: str = "prompt", tests_field: str = "tests",
                setup_field: str = "setup") -> list[tuple[str, list[str], str]]:
    """Load (prompt, tests, setup) triples.

    source: 'builtin' (offline toys) · 'mbpp' · any HF dataset id ('owner/name' or 'hf:owner/name:split').
    For an Adaption-exported HF dataset, the default fields (prompt/tests/setup) match its schema;
    override prompt_field/tests_field/setup_field for other layouts.
    """
    if source == "builtin":
        return _BUILTIN_TASKS
    if source == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test")  # 500 tasks, each with asserts
        prompt_field, tests_field, setup_field = "text", "test_list", "test_setup_code"
    else:
        ds_id, _, split = source.removeprefix("hf:").partition(":")  # 'hf:owner/name:split' or 'owner/name'
        ds = load_dataset(ds_id, split=split or "train")
    end = len(ds) if not n_tasks else min(start + n_tasks, len(ds))
    tasks = []
    for r in ds.select(range(start, end)):
        prompt = (r.get(prompt_field) or "").strip()
        tasks.append((prompt + _REPLY, list(r.get(tests_field) or []), r.get(setup_field) or ""))
    return tasks


def _text(message) -> str:
    """Get text from a str, a dict, or a verifiers pydantic Message (which carries `.content`)."""
    if isinstance(message, str):
        return message
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal content parts
        return "".join(p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") for p in content)
    return ""


class CodeRepairEnv(vf.MultiTurnEnv):
    def __init__(self, tasks, *, max_turns: int, timeout: float, fn: str, efficiency_weight: float, **kwargs):
        self._timeout = timeout
        rows = [{"question": p, "answer": "", "info": {"tests": t, "setup": s}} for p, t, s in tasks]
        ds = Dataset.from_list(rows)
        rubric = vf.Rubric(
            funcs=[self._reward_fn(fn, max_turns, efficiency_weight), self._success_fn(max_turns)],
            weights=[1.0, 0.0],  # success is a logged metric, not scored
        )
        # dataset → RL training (prime train reads get_dataset()); eval_dataset → probe (prime eval run)
        super().__init__(dataset=ds, eval_dataset=ds, rubric=rubric,
                         max_turns=max_turns, message_type="chat", **kwargs)

    @staticmethod
    def _rollout_state(state, max_turns: int) -> RolloutState:
        return RolloutState(
            tests_passed=int(state.get("tests_passed", 0)),
            tests_total=int(state.get("tests_total", 0)),
            turns=int(state.get("turn", 0)),
            max_turns=max_turns,
            succeeded=bool(state.get("solved", False)),
        )

    def _reward_fn(self, fn: str, max_turns: int, efficiency_weight: float):
        def reward(state, **_) -> float:
            s = self._rollout_state(state, max_turns)
            return binary(s) if fn == "binary" else shaped(s, efficiency_weight)
        return reward

    def _success_fn(self, max_turns: int):
        def success(state, **_) -> float:
            return binary(self._rollout_state(state, max_turns))
        return success

    async def setup_state(self, state) -> None:
        state["solved"] = False
        state["tests_passed"] = 0
        state["tests_total"] = len(state["info"]["tests"])

    @vf.stop
    async def is_solved(self, state) -> bool:
        return bool(state.get("solved", False))

    async def env_response(self, messages, state, **kwargs):
        info = state["info"]
        setup, tests = info.get("setup", ""), info["tests"]
        code = extract_code(_text(messages[-1]))
        program = f"{setup}\n{code}" if setup else code
        failed = [t for t in tests if score_code(program, [t], self._timeout)[0] == 0]
        state["tests_passed"] = len(tests) - len(failed)
        state["tests_total"] = len(tests)
        if not failed:
            state["solved"] = True
            return [{"role": "user", "content": "All tests passed. Done."}]
        feedback = "These tests still fail:\n" + "\n".join(failed) + \
                   "\nFix the function and reply with a corrected ```python block."
        return [{"role": "user", "content": feedback}]


def load_environment(source: str = "mbpp", n_tasks: int | None = 20, start: int = 0,
                     prompt_field: str = "prompt", tests_field: str = "tests", setup_field: str = "setup",
                     max_turns: int = 5, timeout: float = 5.0,
                     fn: str = "shaped", efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    tasks = _load_tasks(source, n_tasks, start, prompt_field, tests_field, setup_field)
    return CodeRepairEnv(tasks, max_turns=max_turns, timeout=timeout,
                         fn=fn, efficiency_weight=efficiency_weight)
