import pytest
from scaffold_emit import emit_call as _emit

from laguna_rlvr.scaffold import FORMATS, Tool, parse_call, render_instructions, resolve_format

_TOOLS = [Tool("ocr", "image_id", "read an image"), Tool("answer", "value", "submit the final answer")]


class TestRoundTrip:
    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("name,arg,value", [("ocr", "image_id", "invoice.png"),
                                                ("answer", "value", "42.50")])
    def test_parse_recovers_emitted_call(self, fmt, name, arg, value):
        assert parse_call(_emit(fmt, name, arg, value), fmt, _TOOLS) == (name, value)

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_render_names_every_tool(self, fmt):
        text = render_instructions(_TOOLS, fmt)
        assert "ocr" in text and "answer" in text


class TestParseRobustness:
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_none_when_no_call(self, fmt):
        assert parse_call("just prose, no tool here", fmt, _TOOLS) is None

    def test_line_ignores_unknown_tool(self):
        assert parse_call("foo: bar", "line", _TOOLS) is None

    def test_json_amid_prose(self):
        assert parse_call('sure: {"tool":"ocr","image_id":"a.png"} done', "json", _TOOLS) == ("ocr", "a.png")

    def test_xml_tolerates_reasoning_prefix(self):
        msg = 'let me read it first\n<tool_call>{"name":"answer","arguments":{"value":"7"}}</tool_call>'
        assert parse_call(msg, "xml", _TOOLS) == ("answer", "7")

    def test_strips_trailing_punctuation(self):
        assert parse_call("ocr: invoice.png.", "line", _TOOLS) == ("ocr", "invoice.png")


class TestResolveFormat:
    def test_fixed_passthrough(self):
        assert resolve_format(5, "json") == "json"

    def test_mixed_round_robins_over_all_formats(self):
        assert {resolve_format(i, "mixed") for i in range(len(FORMATS))} == set(FORMATS)

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            resolve_format(0, "bogus")
