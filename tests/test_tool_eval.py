import pytest

from laguna_rlvr.visual.tool_eval import accuracy_matrix, assemble_prompt
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
