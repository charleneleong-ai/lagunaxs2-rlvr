import pytest

from laguna_rlvr.visual.tool_eval import accuracy_matrix, assemble_prompt, rouge_l_f1
from laguna_rlvr.visual.model import IMAGE_TOKEN


class TestAssemblePrompt:
    def test_encoder_splices_image_no_transcript(self):
        p = assemble_prompt("encoder", "TOTAL DUE 42.50", "what is the total?")
        assert IMAGE_TOKEN in p and "TOTAL DUE 42.50" not in p  # encoder reads pixels, not the transcript

    def test_tool_reads_transcript_no_image(self):
        p = assemble_prompt("tool", "TOTAL DUE 42.50", "what is the total?")
        assert "TOTAL DUE 42.50" in p and IMAGE_TOKEN not in p

    def test_encoder_tool_has_both(self):
        p = assemble_prompt("encoder_tool", "TOTAL DUE 42.50", "what is the total?")
        assert "TOTAL DUE 42.50" in p and IMAGE_TOKEN in p
        assert p.index("TOTAL DUE 42.50") < p.index(IMAGE_TOKEN)  # transcript precedes the splice

    def test_rejects_unknown_config(self):
        with pytest.raises(ValueError):
            assemble_prompt("bogus", "t", "q")


class TestAccuracyMatrix:
    def test_per_corpus_and_overall_microaverage(self):
        hits = {"tool": {"docvqa": [3, 4], "vqav2": [0, 6]}}
        m = accuracy_matrix(hits)["tool"]
        assert m["docvqa"] == 0.75 and m["vqav2"] == 0.0
        assert m["overall"] == pytest.approx(3 / 10)  # micro-avg over items, not macro over corpora

    def test_empty_is_zero_not_crash(self):
        assert accuracy_matrix({"encoder": {}})["encoder"] == {"overall": 0.0}


class TestRougeL:
    def test_identical_is_one(self):
        assert rouge_l_f1("crude birth rate in France", "crude birth rate in France") == 1.0

    def test_disjoint_is_zero(self):
        assert rouge_l_f1("crude birth rate", "social media platforms") == 0.0

    def test_partial_overlap_credits_subsequence(self):
        # the metric artifact this fixes: right domain, wrong specifics — exact match scores 0, ROUGE-L > 0
        score = rouge_l_f1("crude birth rate in France 1800", "crude birth rate in Germany 1805")
        assert 0.0 < score < 1.0

    @pytest.mark.parametrize("gold,pred", [("", "x"), ("x", "")])
    def test_empty_side_is_zero(self, gold, pred):
        assert rouge_l_f1(gold, pred) == 0.0
