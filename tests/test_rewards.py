import pytest

from laguna_finetune.rewards import (
    RolloutState,
    binary,
    diff_ratio,
    efficiency_bonus,
    make_scorer,
    partial_credit,
    agreement_score,
    shaped,
    token_efficiency,
)


def _state(passed=5, total=10, turns=10, max_turns=50, succeeded=True) -> RolloutState:
    return RolloutState(passed, total, turns, max_turns, succeeded)


class TestPartialCredit:
    @pytest.mark.parametrize("passed,total,expected", [(0, 10, 0.0), (5, 10, 0.5), (10, 10, 1.0), (3, 0, 0.0)])
    def test_fraction(self, passed, total, expected):
        assert partial_credit(_state(passed, total)) == expected

    def test_monotonic_in_passed(self):
        vals = [partial_credit(_state(p, 10)) for p in range(11)]
        assert vals == sorted(vals)


class TestEfficiencyBonus:
    @pytest.mark.parametrize("turns,max_turns,expected", [(0, 50, 1.0), (50, 50, 0.0), (25, 50, 0.5)])
    def test_value(self, turns, max_turns, expected):
        assert efficiency_bonus(_state(turns=turns, max_turns=max_turns)) == pytest.approx(expected)

    def test_bounded_unit_interval(self):
        assert all(0.0 <= efficiency_bonus(_state(turns=t, max_turns=50)) <= 1.0 for t in range(0, 100, 7))

    def test_decreasing_in_turns(self):
        vals = [efficiency_bonus(_state(turns=t, max_turns=50)) for t in range(0, 51, 5)]
        assert vals == sorted(vals, reverse=True)

    @pytest.mark.parametrize("kw", [{"succeeded": False}, {"max_turns": 0}])
    def test_zero_when_no_signal(self, kw):
        assert efficiency_bonus(_state(**kw)) == 0.0


class TestShaped:
    def test_combines_partial_and_efficiency_on_success(self):
        s = _state(passed=10, total=10, turns=0, max_turns=50, succeeded=True)
        assert shaped(s, efficiency_weight=0.1) == pytest.approx(1.0 + 0.1)

    def test_failure_floors_to_partial_credit(self):
        assert shaped(_state(passed=4, total=10, succeeded=False)) == pytest.approx(0.4)


@pytest.mark.parametrize("succeeded,expected", [(True, 1.0), (False, 0.0)])
def test_binary(succeeded, expected):
    assert binary(_state(succeeded=succeeded)) == expected


class TestTokenEfficiency:
    @pytest.mark.parametrize("used,budget,expected", [(0, 100, 1.0), (50, 100, 0.5), (100, 100, 0.0), (150, 100, 0.0)])
    def test_value(self, used, budget, expected):
        assert token_efficiency(used, budget) == pytest.approx(expected)

    def test_zero_budget_is_zero(self):
        assert token_efficiency(10, 0) == 0.0


class TestDiffRatio:
    def test_identical_is_one(self):
        assert diff_ratio("def f(): return 1", "def f(): return 1") == 1.0

    def test_both_empty_is_one(self):
        assert diff_ratio("", "") == 1.0

    def test_small_edit_scores_higher_than_rewrite(self):
        base = "def f(x): return x + 1"
        small = diff_ratio(base, "def f(x): return x - 1")     # one-char fix
        rewrite = diff_ratio(base, "class Q:\n    pass\nlambda: 0")
        assert small > rewrite


class TestAgreementScore:
    def test_too_few_tests_is_zero(self):
        assert agreement_score(self_passed=1, self_total=1, hidden_pass_fraction=1.0, min_tests=2) == 0.0

    def test_perfect_match_is_one(self):
        assert agreement_score(3, 3, hidden_pass_fraction=1.0) == 1.0

    def test_self_says_pass_but_hidden_fails(self):
        assert agreement_score(3, 3, hidden_pass_fraction=0.0) == 0.0

    def test_partial_match(self):
        assert agreement_score(1, 2, hidden_pass_fraction=0.5) == pytest.approx(1.0)


class TestMakeScorer:
    def test_binary_fn_ignores_partial(self):
        assert make_scorer("binary")(_state(passed=5, total=10, succeeded=True)) == 1.0

    def test_shaped_fn_uses_partial_plus_efficiency(self):
        s = _state(passed=5, total=10, turns=0, max_turns=50, succeeded=True)
        assert make_scorer("shaped", efficiency_weight=0.1)(s) == pytest.approx(0.5 + 0.1)
