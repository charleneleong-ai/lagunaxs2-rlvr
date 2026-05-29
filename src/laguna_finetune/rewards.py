"""Pure reward-shaping functions. No I/O — imported by both envs and the report."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RolloutState:
    """A single rollout's outcome, normalized for shaping."""

    tests_passed: int
    tests_total: int
    turns: int
    max_turns: int
    succeeded: bool

    @classmethod
    def from_state(cls, state, max_turns: int) -> "RolloutState":
        # TODO(event): the single integration contract — map the live verifiers State
        # keys here once a harness surfaces per-test results; both envs go through this.
        return cls(
            tests_passed=int(state.get("tests_passed", 0)),
            tests_total=int(state.get("tests_total", 0)),
            turns=int(state.get("turn", 0)),
            max_turns=max_turns,
            succeeded=bool(state.get("succeeded", False)),
        )


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


def make_scorer(fn: str, max_turns: int, efficiency_weight: float = 0.1):
    """Build a verifiers reward func from a config-selected shaping fn. Pure — no verifiers import."""
    def score(state) -> float:
        s = RolloutState.from_state(state, max_turns)
        return binary(s) if fn == "binary" else shaped(s, efficiency_weight)
    return score
