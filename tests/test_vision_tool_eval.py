import pytest

from laguna_rlvr.visual.vision_tool_eval import _interpret
from scaffold_emit import emit_call


class TestInterpret:
    """The pure loop step — answer terminates with a solved verdict; a tool call continues with an
    observation. Adapter-free, so the agentic control is tested without the GPU."""

    @pytest.mark.parametrize("fmt", ["line", "xml", "json", "poolside"])
    def test_correct_answer_terminates_solved(self, fmt):
        reply = emit_call(fmt, "answer", "value", "42.50")
        assert _interpret(reply, fmt, gold="42.50", transcript="x") == ("done", True)

    @pytest.mark.parametrize("fmt", ["line", "xml", "json", "poolside"])
    def test_wrong_answer_terminates_unsolved(self, fmt):
        reply = emit_call(fmt, "answer", "value", "99.99")
        assert _interpret(reply, fmt, gold="42.50", transcript="x") == ("done", False)

    @pytest.mark.parametrize("fmt", ["line", "xml", "json", "poolside"])
    def test_ocr_call_serves_the_transcript(self, fmt):
        reply = emit_call(fmt, "ocr", "image_id", "docvqa.png")
        kind, obs = _interpret(reply, fmt, gold="42.50", transcript="Total Due 42.50")
        assert kind == "continue" and "Total Due 42.50" in obs

    def test_unparseable_reprompts_with_instructions(self):
        kind, obs = _interpret("hmm let me think about it", "poolside", gold="42.50", transcript="x")
        assert kind == "continue" and "ocr" in obs.lower() and "answer" in obs.lower()
