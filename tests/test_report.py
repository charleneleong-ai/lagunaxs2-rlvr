import pandas as pd

from laguna_rlvr.report import rank, render_markdown


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["env", "model", "success", "reward"])


class TestRank:
    def test_ranks_by_success_spread_descending(self):
        df = _df([
            {"env": "a", "model": "m", "success": True, "reward": 1.0},
            {"env": "a", "model": "m", "success": False, "reward": 0.0},   # base .5 -> signal .25
            {"env": "b", "model": "m", "success": True, "reward": 0.9},
            {"env": "b", "model": "m", "success": True, "reward": 0.7},    # base 1.0 -> signal 0
        ])
        assert [r.env for r in rank(df)] == ["a", "b"]

    def test_saturated_domain_with_reward_jitter_reads_as_flat(self):
        # all solved (base_rate 1.0) but rewards differ (efficiency jitter) -> reward variance > 0,
        # yet no RL gradient. Must be flagged flat (the bug the multilingual seed probe exposed).
        df = _df([
            {"env": "sat", "model": "m", "success": True, "reward": 1.1},
            {"env": "sat", "model": "m", "success": True, "reward": 1.0},
        ])
        r = rank(df)[0]
        assert r.variance > 0.0 and r.signal == 0.0 and r.learnable is False

    def test_zero_variance_flagged_and_ranked_last(self):
        df = _df([
            {"env": "flat", "model": "m", "success": True, "reward": 1.0},
            {"env": "flat", "model": "m", "success": True, "reward": 1.0},   # var 0
            {"env": "live", "model": "m", "success": True, "reward": 1.0},
            {"env": "live", "model": "m", "success": False, "reward": 0.0},  # var .25
        ])
        ranked = rank(df)
        assert ranked[0].env == "live" and ranked[0].learnable is True
        assert ranked[-1].env == "flat" and ranked[-1].learnable is False

    def test_single_record_group_has_zero_variance(self):
        ranked = rank(_df([{"env": "solo", "model": "m", "success": True, "reward": 1.0}]))
        assert ranked[0].variance == 0.0 and ranked[0].learnable is False


def test_render_markdown_emits_a_row_per_group():
    df = _df([{"env": "a", "model": "m", "success": True, "reward": 1.0},
              {"env": "a", "model": "m", "success": False, "reward": 0.0}])
    assert "| a | m |" in render_markdown(rank(df))
