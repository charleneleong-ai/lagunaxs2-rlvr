"""Pure reward-shaping functions. No I/O — imported by both envs and the report."""
from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RolloutState:
    """A single rollout's outcome, normalized for shaping. Each env maps its own harness
    State keys to this (the keys differ per harness, so the mapping lives in the env)."""

    tests_passed: int
    tests_total: int
    turns: int
    max_turns: int
    succeeded: bool


def binary(s: RolloutState) -> float:
    return float(s.succeeded)


def partial_credit(s: RolloutState) -> float:
    if s.tests_total <= 0:
        return 0.0
    return s.tests_passed / s.tests_total


def efficiency_bonus(s: RolloutState) -> float:
    """Bounded [0, 1], decreasing in turns. Zero unless the rollout succeeded — never pay for fast failure."""
    if not s.succeeded or s.max_turns <= 0:
        return 0.0
    return max(0.0, 1.0 - s.turns / s.max_turns)


def shaped(s: RolloutState, efficiency_weight: float = 0.1) -> float:
    """Dense signal for a small model: test-pass fraction plus an efficiency nudge on success."""
    return partial_credit(s) + efficiency_weight * efficiency_bonus(s)


def make_scorer(fn: str, efficiency_weight: float = 0.1):
    """Select a shaping fn by config name. Operates on a RolloutState the env has built."""
    def score(s: RolloutState) -> float:
        return binary(s) if fn == "binary" else shaped(s, efficiency_weight)
    return score


# --- composite reward components (efficiency / minimal-diff / self-verify) ---
# Each is pure and bounded [0, 1]; the env weights them into a verifiers Rubric.

def token_efficiency(tokens_used: int, budget: int) -> float:
    """1.0 at zero tokens, → 0 at/over budget. Pairs with turn-based efficiency_bonus."""
    if budget <= 0:
        return 0.0
    return max(0.0, 1.0 - tokens_used / budget)


def diff_ratio(before: str, after: str) -> float:
    """Similarity of two code strings, [0, 1]. 1.0 = identical (a minimal/surgical edit).

    Reward small fixes: high when `after` barely changes the buggy `before`.
    """
    if not before and not after:
        return 1.0
    return difflib.SequenceMatcher(None, before, after).ratio()


def agreement_score(self_passed: int, self_total: int, hidden_pass_fraction: float,
                    min_tests: int = 2) -> float:
    """How well the model's OWN tests track the hidden outcome, [0, 1].

    Zero if it wrote fewer than `min_tests` (discourage trivial/no tests); otherwise
    1 - |self_pass_fraction - hidden_pass_fraction| (its self-assessment matches reality).
    """
    if self_total < min_tests:
        return 0.0
    self_fraction = self_passed / self_total
    return max(0.0, 1.0 - abs(self_fraction - hidden_pass_fraction))
