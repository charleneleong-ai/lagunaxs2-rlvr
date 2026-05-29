"""Multi-turn, no-Docker code-repair env: write a function, run its tests, fix on failures, repeat.

Tasks come from MBPP (a real difficulty spread, so a strong model's pass rate is intermediate ->
a learnable RL gradient), with a small `builtin` set for offline/tests. Reward is dense shaped
partial credit (test-pass fraction + an efficiency nudge for solving in fewer turns).

verifiers 0.1.14 MultiTurnEnv API: override setup_state (mutate in place) and env_response
(-> Messages); completion via the @vf.stop method; scoring via the rubric. No sandbox — tests
run in a local subprocess (code_exec.score_code), so it's $0 with any OpenAI-compatible model.
"""
from __future__ import annotations

import verifiers as vf
from datasets import Dataset, load_dataset

from laguna_rlvr.code_exec import extract_code, message_text, score_code   # vendor before any Hub push
from laguna_rlvr.rewards import RolloutState, binary, shaped

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


class CodeRepairEnv(vf.MultiTurnEnv):
    def __init__(self, tasks, *, max_turns: int, timeout: float, fn: str, efficiency_weight: float, **kwargs):
        self._timeout = timeout
        rows = [{"question": p, "answer": "", "info": {"tests": t, "setup": s}} for p, t, s in tasks]
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
        info = state["info"]
        setup, tests = info.get("setup", ""), info["tests"]
        code = extract_code(message_text(messages[-1]))
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
