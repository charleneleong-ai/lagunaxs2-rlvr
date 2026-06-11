import pytest

from laguna_rlvr.visual.ocr_backend_eval import cer, coverage_matrix, wer


class TestWER:
    @pytest.mark.parametrize("ref,hyp,expected", [
        ("the quick brown fox", "the quick brown fox", 0.0),       # identical
        ("the quick brown fox", "the quick green fox", 0.25),      # 1 substitution / 4
        ("the quick brown fox", "the quick brown", 0.25),          # 1 deletion / 4
        ("the quick brown fox", "", 1.0),                          # all deleted
        ("", "", 0.0),                                             # empty both
        ("", "spurious text", 1.0),                                # all insertions
        ("The  Quick  FOX", "the quick fox", 0.0),                 # normalized: case + whitespace
    ])
    def test_wer(self, ref, hyp, expected):
        assert wer(ref, hyp) == pytest.approx(expected)


class TestCER:
    @pytest.mark.parametrize("ref,hyp,expected", [
        ("kitten", "sitting", 3 / 6),    # canonical Levenshtein: k->s, e->i, +g
        ("total", "total", 0.0),
        ("total", "", 1.0),
    ])
    def test_cer(self, ref, hyp, expected):
        assert cer(ref, hyp) == pytest.approx(expected)


class TestCoverage:
    def test_per_corpus_and_overall(self):
        rows = [
            {"corpus": "textvqa", "gold": "nokia", "transcript": "the sign reads NOKIA store"},   # hit (case)
            {"corpus": "textvqa", "gold": "sony", "transcript": "samsung galaxy ad"},             # miss
            {"corpus": "docvqa", "gold": "42.50", "transcript": "Total Due: $42.50"},             # hit
        ]
        cov = coverage_matrix(rows)
        assert cov["textvqa"] == pytest.approx(0.5)
        assert cov["docvqa"] == pytest.approx(1.0)
        assert cov["overall"] == pytest.approx(2 / 3)

    def test_single_token_needs_word_boundary(self):
        # `_match` semantics: a single-token gold must not be credited when glued inside a longer run
        # (digit-spam mirage) — coverage inherits that, so a backend can't fake coverage with noise.
        glued = coverage_matrix([{"corpus": "dvqa", "gold": "2", "transcript": "20000 30000 values"}])
        clean = coverage_matrix([{"corpus": "dvqa", "gold": "2", "transcript": "the bar shows 2 units"}])
        assert glued["overall"] == 0.0 and clean["overall"] == 1.0
