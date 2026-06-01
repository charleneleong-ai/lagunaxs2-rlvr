import pytest

from laguna_rlvr.visual.mmmu import (
    _free_match,
    _match_choice,
    _norm_answer,
    _option_lines,
    _parse_choice,
    _parse_options,
    mathvista_eval,
    mmmu_eval,
)


def test_option_lines_handles_more_than_seven_options():
    # MMMU has questions with >7 options; _LETTERS must not overflow (the n=64 IndexError)
    lines = _option_lines([f"opt{i}" for i in range(9)])
    assert "\nA. opt0" in lines and "\nI. opt8" in lines  # letters extend past G to I


def test_parse_choice_supports_late_letters():
    assert _parse_choice("the answer is I", 9) == "I"


class _FakeAdapter:
    """Ignores inputs, returns scripted replies in order — drives the eval aggregation offline."""

    def __init__(self, replies: list[str]):
        self._replies = iter(replies)

    def chat(self, turns, max_new_tokens: int = 64) -> list[str]:
        return [next(self._replies)]


class TestParseChoice:
    @pytest.mark.parametrize("reply,letter", [
        ("A", "A"), ("(B)", "B"), ("B.", "B"), ("The answer is C", "C"),
        ("I think the answer is option D.", "D"),
    ])
    def test_finds_letter(self, reply, letter):
        assert _parse_choice(reply, 5) == letter

    def test_none_when_absent(self):
        assert _parse_choice("no idea", 5) is None

    def test_out_of_range_letter_ignored(self):
        assert _parse_choice("E", 3) is None  # only A-C valid for 3 options


class TestMatchChoice:
    CHOICES = ["red apple", "green pear", "blue plum"]

    @pytest.mark.parametrize("reply,expected", [
        ("B", "green pear"),                      # by letter
        ("The answer is a blue plum.", "blue plum"),  # by substring
        ("(A)", "red apple"),
    ])
    def test_matches(self, reply, expected):
        assert _match_choice(reply, self.CHOICES) == expected

    def test_none_on_miss(self):
        assert _match_choice("an orange", self.CHOICES) is None


class TestNormAnswer:
    @pytest.mark.parametrize("reply,gold", [
        ("3.0", "3"), ("$3", "3"), ("the result is 42", "42"), ("3.005", "3"),
    ])
    def test_numeric_tolerance(self, reply, gold):
        assert _free_match(reply, gold)

    @pytest.mark.parametrize("reply,gold", [("5", "3"), ("3.5", "3")])
    def test_numeric_miss(self, reply, gold):
        assert not _free_match(reply, gold)

    def test_string_normalization(self):
        assert _free_match("The answer: Triangle.", "triangle")
        assert _norm_answer("  $Triangle.! ") == "triangle"


def test_parse_options_stringified_list():
    assert _parse_options('["x", "y", "z"]') == ["x", "y", "z"]
    assert _parse_options(["a", "b"]) == ["a", "b"]
    assert _parse_options(None) == [] and _parse_options("garbage") == []


def test_mmmu_eval_accuracy():
    items = [
        # (image, question_with_options, gold_letter, question_type)
        (None, "Q?\nA. cat\nB. dog\nC. fish", "B", "multiple-choice"),
        (None, "Q?\nA. cat\nB. dog\nC. fish", "A", "multiple-choice"),
        (None, "Name the shape", "triangle", "open"),
    ]
    adapter = _FakeAdapter(["The answer is B", "C", "It is a triangle"])
    out = mmmu_eval(adapter, items)
    # B correct, C != A wrong, triangle substring correct -> 2/3
    assert out == {"mmmu/metrics/accuracy": pytest.approx(2 / 3)}


def test_mathvista_eval_accuracy():
    items = [
        # (image, prompt, answer, question_type, choices)
        (None, "pick one", "8", "multi_choice", ["4", "8", "12"]),
        (None, "pick one", "12", "multi_choice", ["4", "8", "12"]),
        (None, "compute", "3.0", "free_form", []),
    ]
    adapter = _FakeAdapter(["B", "the value is 8", "approximately 3"])
    out = mathvista_eval(adapter, items)
    # B->"8" correct, "8" substring != "12" wrong, 3≈3.0 correct -> 2/3
    assert out == {"mathvista/metrics/accuracy": pytest.approx(2 / 3)}
