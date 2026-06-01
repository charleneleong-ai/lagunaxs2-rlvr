import pytest
import torch

from laguna_rlvr.visual.gspo import _surrogate_loss, read_reward


@pytest.mark.parametrize("mode", ["gspo", "grpo"])
def test_surrogate_loss_stays_finite_under_exploding_logratio(mode):
    # a huge cur-vs-old logp gap (e.g. MoE routing differs old/new) would overflow exp() to inf
    # without the pre-exp clamp -> NaN grad -> the device-side assert that killed the run.
    cur = torch.zeros(4, 6)
    old = torch.full((4, 6), -1000.0)  # Δlogp = +1000/token
    mask = torch.ones(4, 6)
    adv = torch.tensor([1.0, -1.0, 2.0, -2.0])
    loss = _surrogate_loss(cur, old, mask, adv, mode=mode, clip=0.2, max_logratio=10.0)
    assert torch.isfinite(loss)


def test_surrogate_loss_equal_logp_is_negative_mean_advantage():
    # cur == old -> importance ratio s = 1 -> pg = adv -> loss = -mean(adv)
    lp = torch.randn(3, 4)
    adv = torch.tensor([1.0, 2.0, 3.0])
    loss = _surrogate_loss(lp, lp, torch.ones(3, 4), adv, mode="gspo", clip=0.2, max_logratio=10.0)
    assert torch.isclose(loss, -adv.mean(), atol=1e-5)


@pytest.mark.parametrize("needle,completion,expected", [
    ("alpha", "here is alpha ok", 1.0),          # substring / strong overlap -> exact reward
    ("alpha beta", "x y z", 0.0),                 # disjoint -> no credit
])
def test_read_reward_endpoints(needle, completion, expected):
    assert read_reward(needle, completion) == expected


def test_read_reward_partial_is_bounded_below_exact():
    # weak (sub-_match) overlap earns partial credit in (0, 0.5] so groups get gradient variance
    r = read_reward("alpha beta gamma delta", "gamma epsilon")
    assert 0.0 < r <= 0.5
