import pytest

from laguna_rlvr.visual.ocrbench import _HME, _ocrbench_correct, ocrbench_eval


class TestOCRBenchCorrect:
    """The per-item OCRBench match protocol: substring (case-insensitive) + HME whitespace-exact."""

    @pytest.mark.parametrize("pred,answers,category,expected", [
        ("The total is 42 dollars", ["42"], "Scene Text-centric VQA", True),   # substring hit
        ("HELLO World", ["hello"], "Text Recognition", True),                  # case-insensitive
        ("the price is 99", ["42"], "Scene Text-centric VQA", False),          # miss
        ("answer: cat", ["dog", "cat"], "KIE", True),                          # any gold answer hits
        (r"x = \frac{1}{2}", [r"x=\frac{1}{2}"], _HME, True),                  # HME whitespace-exact
        (r"x = \frac{1}{2} + 1", [r"x=\frac{1}{2}"], _HME, False),             # HME not substring-lenient
    ])
    def test_match(self, pred, answers, category, expected):
        assert _ocrbench_correct(pred, answers, category) is expected


class _FakeAdapter:
    """Stand-in returning canned replies in order, ignoring the prompt/image."""

    def __init__(self, replies: list[str]):
        self._replies = iter(replies)

    def chat(self, turns, max_new_tokens):
        return [next(self._replies)]


class TestOCRBenchEval:
    def test_overall_and_per_category_accuracy(self):
        items = [
            (None, "q", ["cat"], "Text Recognition"),          # pred hits
            (None, "q", ["dog"], "Text Recognition"),          # pred misses
            (None, "q", [r"x^2"], _HME),                       # HME hits (whitespace-exact)
        ]
        adapter = _FakeAdapter(["a cat sat", "a bird", r"x ^ 2"])
        out = ocrbench_eval(adapter, items, prefix="ocrbench")
        assert out["ocrbench/metrics/accuracy"] == pytest.approx(2 / 3)
        assert out["ocrbench/metrics/acc_text_recognition"] == pytest.approx(0.5)
        assert out["ocrbench/metrics/acc_handwritten_mathematical_expression_recognition"] == 1.0
