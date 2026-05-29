from laguna_finetune.report import DomainRanking
from laguna_finetune.rl import should_train


def _ranking(env: str, variance: float, base: float = 0.5) -> DomainRanking:
    return DomainRanking(env, "laguna", 10, base, variance, base * variance, variance > 0.0)


class TestShouldTrain:
    def test_refuses_zero_variance(self):
        ok, reason = should_train([_ranking("flat", 0.0)], "flat")
        assert ok is False and "gradient" in reason

    def test_allows_learnable_signal(self):
        ok, _ = should_train([_ranking("live", 0.25)], "live")
        assert ok is True

    def test_refuses_when_env_absent(self):
        ok, reason = should_train([_ranking("live", 0.25)], "absent")
        assert ok is False and "run the probe" in reason
