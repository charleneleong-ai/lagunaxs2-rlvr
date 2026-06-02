import json

import pytest
from scaffold_emit import emit_call as _emit

from laguna_rlvr.scaffold import (FORMATS, Tool, parse_call, parse_native, render_instructions,
                                  resolve_format, to_tool_defs)

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

    def test_native_passes_through_not_in_mixed(self):
        assert resolve_format(0, "native") == "native"
        assert "native" not in FORMATS   # native is an env-level mode, not a round-robined text format

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            resolve_format(0, "bogus")


class TestNative:
    def test_to_tool_defs_shape(self):
        defs = to_tool_defs(_TOOLS)   # vf.Tool flat format, not the legacy OpenAI {type,function} wrapper
        assert {d["name"] for d in defs} == {"ocr", "answer"}
        ocr_def = next(d for d in defs if d["name"] == "ocr")
        assert ocr_def["parameters"]["required"] == ["image_id"]

    def test_parse_native_dict_message(self):
        msg = {"tool_calls": [{"function": {"name": "ocr", "arguments": '{"image_id": "a.png"}'}}]}
        assert parse_native(msg, _TOOLS) == ("ocr", "a.png")

    def test_parse_native_flat_json_string_call(self):
        # the shape verifiers/prime actually returns: a JSON string, name/arguments at top level
        call = json.dumps({"id": "x", "name": "ocr", "arguments": json.dumps({"image_id": "a.png"})})
        assert parse_native({"tool_calls": [call]}, _TOOLS) == ("ocr", "a.png")

    def test_parse_native_object_message_with_dict_args(self):
        fn = type("Fn", (), {"name": "answer", "arguments": {"value": "42"}})()
        msg = type("Msg", (), {"tool_calls": [type("Call", (), {"function": fn})()]})()
        assert parse_native(msg, _TOOLS) == ("answer", "42")

    @pytest.mark.parametrize("msg", [
        {"tool_calls": None},                                                          # no calls
        {"tool_calls": [{"function": {"name": "unknown", "arguments": "{}"}}]},         # unknown tool
        {"content": "just text"},                                                      # not a tool message
    ])
    def test_parse_native_none(self, msg):
        assert parse_native(msg, _TOOLS) is None
