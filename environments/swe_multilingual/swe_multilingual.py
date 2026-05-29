"""Per-language SWE tasks targeting poolside's reported multilingual headroom (57.7% vs 69.9% Verified)."""
import verifiers as vf

from laguna_finetune.rewards import RolloutState, binary, make_scorer


def load_environment(fn: str = "shaped", efficiency_weight: float = 0.1,
                     langs: list[str] | None = None, max_turns: int = 50, **kwargs) -> vf.Environment:
    score = make_scorer(fn, max_turns, efficiency_weight)

    def reward(state, **_) -> float:
        return score(state)

    def success(state, **_) -> float:
        return binary(RolloutState.from_state(state, max_turns))

    rubric = vf.Rubric(funcs=[reward, success], weights=[1.0, 0.0])  # success logged as metric, not scored
    return vf.ToolEnv(dataset=_load_multilingual(langs or ["go", "rust", "typescript"]),
                      rubric=rubric, max_turns=max_turns)


def _load_multilingual(langs: list[str]):
    raise NotImplementedError(
        f"Wire per-language SWE tasks for {langs} here (e.g. SWE-bench Multilingual split). "
        "Probe Laguna's per-language base rate to pick the target language(s).")
