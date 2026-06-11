import pytest

from laguna_rlvr.scaffold import format_call, parse_call
from laguna_rlvr.visual.model import IMAGE_TOKEN
from laguna_rlvr.visual.vision_tool_eval import _TOOLS
from laguna_rlvr.visual.vision_tool_sft import synth_trace


class TestFormatCall:
    @pytest.mark.parametrize("fmt", ["line", "xml", "json", "poolside"])
    @pytest.mark.parametrize("name,arg,value", [("ocr", "image_id", "docvqa.png"), ("answer", "value", "42.50")])
    def test_round_trips_through_parse(self, fmt, name, arg, value):
        # the production emitter must produce exactly what parse_call recovers — else we'd SFT a format the
        # env can't read. Cross-checked against the independent parse implementation.
        assert parse_call(format_call(name, arg, value, fmt), fmt, _TOOLS) == (name, value)


class TestSynthTrace:
    def _supervised(self, segs):
        return [t for t, sup in segs if sup]

    @pytest.mark.parametrize("fmt", ["line", "xml", "json", "poolside"])
    def test_ocr_trace_supervises_ocr_then_answer(self, fmt):
        segs = synth_trace("docvqa.png", "total?", "42.50", "Total Due 42.50", fmt, use_ocr=True)
        assert len(segs) == 4
        sup = self._supervised(segs)
        assert parse_call(sup[0], fmt, _TOOLS) == ("ocr", "docvqa.png")
        assert parse_call(sup[1], fmt, _TOOLS) == ("answer", "42.50")

    def test_direct_trace_supervises_only_the_answer(self):
        segs = synth_trace("dvqa.png", "largest bar?", "soil", "x", "poolside", use_ocr=False)
        assert len(segs) == 2
        sup = self._supervised(segs)
        assert len(sup) == 1 and parse_call(sup[0], "poolside", _TOOLS) == ("answer", "soil")

    def test_prompt_segment_has_image_and_is_unsupervised(self):
        (prompt, supervised), *_ = synth_trace("d.png", "q?", "a", "t", "poolside", use_ocr=True)
        assert IMAGE_TOKEN in prompt and supervised is False
