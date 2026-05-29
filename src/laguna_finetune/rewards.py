"""Pure reward-shaping functions. No I/O — imported by both envs and the report."""
from __future__ import annotations

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
