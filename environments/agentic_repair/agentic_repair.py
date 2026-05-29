"""Composite-reward, long-horizon agentic code REPAIR (bug-fix framing).

Tier-1 substrate: MBPP with a synthetically injected bug in the reference solution, so the
agent does a *surgical fix* over multiple turns. The reward is a reusable, benchmark-agnostic
composite (weighted verifiers Rubric):
  correctness  — hidden-test pass fraction
  efficiency   — solve in fewer turns                                  (#1)
  minimal_diff — stay close to the buggy original (surgical edit)      (#3)
  self_verify  — model's own tests agree with the hidden outcome       (#2, opt-in via write_tests)
The same rubric is meant to drop onto a real SWE env (mini-swe-agent-plus) for the tier-2
long-horizon result. No sandbox — tests run in a local subprocess, so it's $0 with any model.
"""
from __future__ import annotations

import verifiers as vf
from datasets import Dataset, load_dataset

from laguna_finetune.code_exec import extract_code, score_code   # vendor before any Hub push
from laguna_finetune.rewards import RolloutState, agreement_score, diff_ratio, efficiency_bonus

# Deterministic single-edit mutations that tend to break behavior (no RNG — reproducible).
_MUTATIONS = [("==", "!="), ("<=", ">"), (">=", "<"), (" < ", " > "), (" > ", " < "),
              (" + ", " - "), (" - ", " + "), (" and ", " or "), (" or ", " and ")]
_REPLY = "\nReply with the corrected function in a ```python code block."

_BUILTIN_TASKS = [  # (prompt, tests, setup, buggy) — offline fallback for tests
    ("Implement `add(a, b)` returning a + b. This implementation is buggy:\n"
     "```python\ndef add(a, b): return a - b\n```" + _REPLY,
     ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"], "", "def add(a, b): return a - b"),
]


def inject_bug(code: str, tests: list[str], setup: str, timeout: float) -> str | None:
    """Return a one-edit buggy variant of `code` that fails >=1 test, or None if none found."""
    for a, b in _MUTATIONS:
        if a in code:
            buggy = code.replace(a, b, 1)
            if buggy != code and score_code(f"{setup}\n{buggy}", tests, timeout)[0] < len(tests):
                return buggy
    return None


def _load_repair_tasks(source: str, n_tasks: int | None, start: int, timeout: float):
    if source == "builtin":
        return _BUILTIN_TASKS
    if source != "mbpp":
        raise ValueError(f"unknown source {source!r} (use 'mbpp' or 'builtin')")
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    tasks, i, want = [], start, (n_tasks or len(ds))
    while len(tasks) < want and i < len(ds):
        r, i = ds[i], i + 1
        code, tests, setup = r.get("code") or "", list(r["test_list"]), r.get("test_setup_code") or ""
        if not code or score_code(f"{setup}\n{code}", tests, timeout)[0] < len(tests):
            continue  # keep only references that pass their own tests
        buggy = inject_bug(code, tests, setup, timeout)
        if buggy is None:
            continue
        spec = (r.get("text") or r.get("prompt") or "").strip()
        tasks.append((f"{spec}\n\nThis implementation is buggy:\n```python\n{buggy}\n```{_REPLY}",
                      tests, setup, buggy))
    return tasks


def _text(message) -> str:
    if isinstance(message, str):
        return message
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") for p in content)
    return ""


_DEFAULT_WEIGHTS = {"correctness": 1.0, "efficiency": 0.1, "minimal_diff": 0.1, "self_verify": 0.0}


class AgenticRepairEnv(vf.MultiTurnEnv):
    def __init__(self, tasks, *, max_turns: int, timeout: float, weights: dict, **kwargs):
        self._timeout = timeout
        rows = [{"question": p, "answer": "", "info": {"tests": t, "setup": s, "buggy": b}}
                for p, t, s, b in tasks]
        w = {**_DEFAULT_WEIGHTS, **(weights or {})}
        funcs = [self._correctness, self._efficiency_fn(max_turns), self._minimal_diff,
                 self._self_verify_fn(), self._success_fn()]
        weight_list = [w["correctness"], w["efficiency"], w["minimal_diff"], w["self_verify"], 0.0]
        super().__init__(eval_dataset=Dataset.from_list(rows),
                         rubric=vf.Rubric(funcs=funcs, weights=weight_list),
                         max_turns=max_turns, message_type="chat", **kwargs)

    # --- reward components (read the state the env mutates each turn) ---
    @staticmethod
    def _correctness(state, **_) -> float:
        total = state.get("tests_total", 0)
        return state.get("tests_passed", 0) / total if total else 0.0

    def _efficiency_fn(self, max_turns: int):
        def efficiency(state, **_) -> float:
            return efficiency_bonus(_RS(state, max_turns))
        return efficiency

    @staticmethod
    def _minimal_diff(state, **_) -> float:
        if not state.get("solved"):
            return 0.0  # only reward a small diff on a correct fix
        return diff_ratio(state["info"]["buggy"], state.get("last_code", ""))

    def _self_verify_fn(self):
        def self_verify(state, **_) -> float:
            sv = state.get("self_verify")
            if not sv:
                return 0.0
            total = state.get("tests_total", 0)
            hidden_frac = state.get("tests_passed", 0) / total if total else 0.0
            return agreement_score(sv["passed"], sv["total"], hidden_frac)
        return self_verify

    @staticmethod
    def _success_fn():
        def success(state, **_) -> float:
            return float(bool(state.get("solved", False)))
        return success

    async def setup_state(self, state) -> None:
        state["solved"] = False
        state["tests_passed"] = 0
        state["tests_total"] = len(state["info"]["tests"])
        state["last_code"] = ""

    @vf.stop
    async def is_solved(self, state) -> bool:
        return bool(state.get("solved", False))

    async def env_response(self, messages, state, **kwargs):
        info = state["info"]
        setup, tests = info.get("setup", ""), info["tests"]
        code = extract_code(_text(messages[-1]))
        state["last_code"] = code
        program = f"{setup}\n{code}" if setup else code
        failed = [t for t in tests if score_code(program, [t], self._timeout)[0] == 0]
        state["tests_passed"] = len(tests) - len(failed)
        state["tests_total"] = len(tests)
        if not failed:
            state["solved"] = True
            return [{"role": "user", "content": "All tests passed. Done."}]
        return [{"role": "user", "content": "These tests still fail:\n" + "\n".join(failed) +
                 "\nMake a minimal correction and reply with a ```python block."}]


def _RS(state, max_turns: int) -> RolloutState:
    return RolloutState(tests_passed=int(state.get("tests_passed", 0)),
                        tests_total=int(state.get("tests_total", 0)),
                        turns=int(state.get("turn", 0)), max_turns=max_turns,
                        succeeded=bool(state.get("solved", False)))


def load_environment(source: str = "mbpp", n_tasks: int | None = 20, start: int = 0,
                     max_turns: int = 8, timeout: float = 5.0,
                     weights: dict | None = None, **kwargs) -> vf.Environment:
    tasks = _load_repair_tasks(source, n_tasks, start, timeout)
    return AgenticRepairEnv(tasks, max_turns=max_turns, timeout=timeout, weights=weights)
