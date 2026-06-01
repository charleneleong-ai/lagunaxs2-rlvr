import pytest

from laguna_rlvr.visual.mmdu import _overlap, mmdu_eval


class _FakeAdapter:
    """`.chat(turns)` returns one scripted reply per turn (ignores the turn content)."""

    def __init__(self, scripts: list[list[str]]):
        self._scripts = iter(scripts)

    def chat(self, turns, max_new_tokens: int = 128) -> list[str]:
        return next(self._scripts)


def _turn(text: str, reference: str) -> dict:
    return {"text": text, "images": [], "reference": reference}


@pytest.mark.parametrize("reply, reference, expected", [
    ("alpha beta gamma", "alpha beta gamma", 1.0),
    ("alpha beta", "gamma delta", 0.0),
    ("alpha beta", "alpha gamma", 0.5),  # 1 shared / (p=1/2, r=1/2) -> F1 0.5
    ("", "", 1.0),
    ("alpha", "", 0.0),
])
def test_overlap_token_f1(reply, reference, expected):
    assert _overlap(reply, reference) == pytest.approx(expected)


def test_mmdu_eval_accuracy_is_mean_per_turn_overlap():
    # one 2-turn episode: turn0 reply exact (1.0), turn1 reply disjoint from its ref (0.0) -> acc 0.5
    episode = [_turn("q1", "red square circle"), _turn("q2", "blue triangle")]
    adapter = _FakeAdapter([["red square circle", "totally unrelated words"]])
    m = mmdu_eval(adapter, [episode])
    assert m["mmdu/metrics/accuracy"] == pytest.approx(0.5)


def test_mmdu_eval_recall_credits_cross_turn_memory():
    # turn1 reply re-surfaces "square" introduced in turn0's reference -> 1/1 later turns hit -> recall 1.0
    episode = [_turn("q1", "red square circle"), _turn("q2", "blue triangle")]
    adapter = _FakeAdapter([["red square circle", "i recall the square earlier"]])
    m = mmdu_eval(adapter, [episode])
    assert m["mmdu/metrics/recall"] == pytest.approx(1.0)


def test_mmdu_eval_recall_falls_back_to_first_reference_overlap():
    # no later reply shares a prior content token -> fall back to final reply vs FIRST reference overlap.
    # final reply "blue square" vs first ref "blue square" -> overlap 1.0 (stopwords none here)
    episode = [_turn("q1", "blue square"), _turn("q2", "green pentagon")]
    adapter = _FakeAdapter([["something else", "blue square"]])
    m = mmdu_eval(adapter, [episode])
    assert m["mmdu/metrics/recall"] == pytest.approx(1.0)


def test_mmdu_eval_aggregates_over_episodes():
    ep1 = [_turn("q1", "alpha beta"), _turn("q2", "gamma delta")]
    ep2 = [_turn("q1", "one two"), _turn("q2", "three four")]
    # ep1: turn accs 1.0, 0.0 -> mean 0.5 ; ep2: 0.0, 1.0 -> mean 0.5 ; overall mean of 4 turns = 0.5
    adapter = _FakeAdapter([["alpha beta", "nope"], ["nope", "three four"]])
    m = mmdu_eval(adapter, [ep1, ep2])
    assert m["mmdu/metrics/accuracy"] == pytest.approx(0.5)
