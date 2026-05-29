"""Difficulty-filtered Terminal-Bench subset with a shaped reward for small models."""
import verifiers as vf

from laguna_finetune.rewards import RolloutState, binary, make_scorer


def load_environment(fn: str = "shaped", efficiency_weight: float = 0.1,
                     split: str = "curated_easy", max_turns: int = 50, **kwargs) -> vf.Environment:
    score = make_scorer(fn, max_turns, efficiency_weight)

    def reward(state, **_) -> float:
        return score(state)

    def success(state, **_) -> float:
        return binary(RolloutState.from_state(state, max_turns))

    rubric = vf.Rubric(funcs=[reward, success], weights=[1.0, 0.0])  # success logged as metric, not scored
    return vf.ToolEnv(dataset=_load_curated_split(split), rubric=rubric, max_turns=max_turns)


def _load_curated_split(split: str):
    raise NotImplementedError(
        f"Wire the Terminal-Bench '{split}' subset here (HF dataset or local tasks). "
        "Probe Laguna's base rate to choose the difficulty cut.")
