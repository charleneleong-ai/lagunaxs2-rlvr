"""Multi-turn, no-Docker code-repair env: write a function, run its tests, fix on failures, repeat.

Dev-friendly smoke env to green the probe->reward->report pipeline locally with any
OpenAI-compatible model (e.g. Ollama), $0, no sandbox. Exercises the agentic loop and the
dense shaped reward (test-pass fraction + an efficiency nudge for solving in fewer turns).

Targets the verifiers 0.1.14 MultiTurnEnv API: override setup_state (mutate in place) and
env_response (-> Messages); completion via the @vf.stop method; scoring via the rubric.
"""
from __future__ import annotations

import verifiers as vf
from datasets import Dataset

from laguna_finetune.code_exec import extract_code, score_code   # vendor before any Hub push
from laguna_finetune.rewards import RolloutState, binary, shaped

_TASKS = [
    ("Write a Python function `add(a, b)` that returns their sum. Reply with a ```python code block.",
     ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]),
    ("Write a Python function `is_palindrome(s)` returning True iff s reads the same backwards. "
     "Reply with a ```python code block.",
     ["assert is_palindrome('racecar')", "assert not is_palindrome('hello')", "assert is_palindrome('')"]),
    ("Write a Python function `fib(n)` returning the n-th Fibonacci number (fib(0)=0, fib(1)=1). "
     "Reply with a ```python code block.",
     ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(10) == 55"]),
]


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
        rows = [{"question": prompt, "answer": "", "info": {"tests": tests}} for prompt, tests in tasks]
        rubric = vf.Rubric(
            funcs=[self._reward_fn(fn, max_turns, efficiency_weight), self._success_fn(max_turns)],
            weights=[1.0, 0.0],  # success is a logged metric, not scored
        )
        super().__init__(eval_dataset=Dataset.from_list(rows), rubric=rubric,
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
        code = extract_code(_text(messages[-1]))
        tests = state["info"]["tests"]
        failed = [t for t in tests if score_code(code, [t], self._timeout)[0] == 0]
        state["tests_passed"] = len(tests) - len(failed)
        state["tests_total"] = len(tests)
        if not failed:
            state["solved"] = True
            return [{"role": "user", "content": "All tests passed. Done."}]
        feedback = "These tests still fail:\n" + "\n".join(failed) + \
                   "\nFix the function and reply with a corrected ```python block."
        return [{"role": "user", "content": feedback}]


def load_environment(max_turns: int = 5, timeout: float = 5.0,
                     fn: str = "shaped", efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    return CodeRepairEnv(_TASKS, max_turns=max_turns, timeout=timeout,
                         fn=fn, efficiency_weight=efficiency_weight)
