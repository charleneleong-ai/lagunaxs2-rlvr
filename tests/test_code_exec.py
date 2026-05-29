import pytest

from laguna_finetune.code_exec import extract_code, score_code


class TestExtractCode:
    def test_pulls_fenced_python_block(self):
        assert extract_code("blah\n```python\nx = 1\n```\ntail") == "x = 1"

    def test_pulls_unlabeled_fence(self):
        assert extract_code("```\ny = 2\n```") == "y = 2"

    def test_falls_back_to_whole_text(self):
        assert extract_code("def f(): return 1") == "def f(): return 1"


class TestScoreCode:
    _TESTS = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]

    def test_all_pass(self):
        assert score_code("def add(a, b): return a + b", self._TESTS) == (3, 3)

    def test_partial(self):
        # off-by-one impl: add(2,3)=6≠5 fails; add(-1,1)=1≠0 fails; add(0,0)=1≠0 fails
        assert score_code("def add(a, b): return a + b + 1", self._TESTS) == (0, 3)

    def test_some_pass(self):
        # returns a only: add(2,3)=2≠5 fail; add(-1,1)=-1≠0 fail; add(0,0)=0==0 pass
        assert score_code("def add(a, b): return a", self._TESTS) == (1, 3)

    def test_broken_code_scores_zero_not_crash(self):
        assert score_code("def add(a, b): syntax error here", self._TESTS) == (0, 3)

    def test_timeout_counts_as_fail(self):
        passed, total = score_code("def add(a, b):\n    while True: pass",
                                   ["assert add(1, 1) == 2"], timeout=1.0)
        assert (passed, total) == (0, 1)
