import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parent.parent / "environments" / "ocr_tool" / "ocr_tool.py"
_spec = importlib.util.spec_from_file_location("ocr_tool", _PATH)
ocr_tool = importlib.util.module_from_spec(_spec)
sys.modules["ocr_tool"] = ocr_tool
_spec.loader.exec_module(ocr_tool)


class TestMatch:
    @pytest.mark.parametrize("gold,guess", [
        ("42.50", "42.50"), ("42.50", "$42.50"), ("42.50", "42.5"),        # currency + float-eq
        ("1234", "****1234"), ("jane@clinic.org", "the email is jane@clinic.org"),  # substring
        ("MX-88231", "mx-88231"),                                          # case-insensitive
    ])
    def test_accepts(self, gold, guess):
        assert ocr_tool._match(gold, guess)

    @pytest.mark.parametrize("gold,guess", [
        ("42.50", "38.00"), ("1234", "5678"), ("jane@clinic.org", "bob@clinic.org"), ("MX-88231", ""),
    ])
    def test_rejects(self, gold, guess):
        assert not ocr_tool._match(gold, guess)


class TestOCRBackend:
    def test_returns_text_for_matching_id(self):
        assert ocr_tool.ocr("invoice.png", {"invoice.png": "Total: $5"}) == "Total: $5"

    def test_rejects_wrong_id(self):
        assert "no image" in ocr_tool.ocr("wrong.png", {"invoice.png": "Total: $5"})


from scaffold_emit import emit_call as _toolcall   # noqa: E402


class TestEnv:
    def test_builtin_rows_hide_answer_and_mix_scaffolds(self):
        env = ocr_tool.load_environment(scaffold="mixed")
        rows = env.eval_dataset.to_list()
        assert len(rows) == len(ocr_tool._BUILTIN_DOCS)
        for row in rows:
            # The answer must NOT be readable from the prompt — it lives only in the OCR-able doc.
            assert ocr_tool._norm(row["info"]["answer"]) not in ocr_tool._norm(row["question"])
            assert row["info"]["text"], "doc text needed for the ocr tool"
        assert len({r["info"]["fmt"] for r in rows}) >= 2, "'mixed' must vary the scaffold across rows"

    @pytest.mark.parametrize("fmt", ["line", "xml", "json", "poolside"])
    def test_ocr_then_answer_solves_in_each_scaffold(self, fmt):
        env = ocr_tool.load_environment(scaffold=fmt)
        state = {"info": {"image_id": "invoice.png", "text": "Total Due: $42.50",
                          "answer": "42.50", "fmt": fmt}, "turn": 1}
        asyncio.run(env.setup_state(state))
        obs = asyncio.run(env.env_response(
            [{"role": "assistant", "content": _toolcall(fmt, "ocr", "image_id", "invoice.png")}], state))
        assert "42.50" in obs[0]["content"] and not state["done"]
        asyncio.run(env.env_response(
            [{"role": "assistant", "content": _toolcall(fmt, "answer", "value", "42.50")}], state))
        assert state["solved"] and state["done"]

    def test_wrong_answer_is_done_but_unsolved(self):
        env = ocr_tool.load_environment(scaffold="line")
        state = {"info": {"image_id": "x.png", "text": "Total: $9", "answer": "9", "fmt": "line"}, "turn": 1}
        asyncio.run(env.setup_state(state))
        asyncio.run(env.env_response([{"role": "assistant", "content": "answer: 42"}], state))
        assert state["done"] and not state["solved"]

    def test_native_scaffold_reads_structured_tool_calls(self):
        env = ocr_tool.load_environment(scaffold="native")
        assert all(r["info"]["fmt"] == "native" for r in env.eval_dataset.to_list())
        state = {"info": {"image_id": "invoice.png", "text": "Total Due: $42.50",
                          "answer": "42.50", "fmt": "native"}, "turn": 1}
        asyncio.run(env.setup_state(state))
        ocr_msg = {"role": "assistant", "content": "",
                   "tool_calls": [{"function": {"name": "ocr", "arguments": '{"image_id":"invoice.png"}'}}]}
        obs = asyncio.run(env.env_response([ocr_msg], state))
        assert "42.50" in obs[0]["content"] and not state["done"]
        ans_msg = {"role": "assistant", "content": "",
                   "tool_calls": [{"function": {"name": "answer", "arguments": '{"value":"42.50"}'}}]}
        asyncio.run(env.env_response([ans_msg], state))
        assert state["solved"] and state["done"]

    def test_unparseable_reprompts_without_terminating(self):
        env = ocr_tool.load_environment(scaffold="line")
        state = {"info": {"image_id": "x.png", "text": "Total: $9", "answer": "9", "fmt": "line"}, "turn": 1}
        asyncio.run(env.setup_state(state))
        obs = asyncio.run(env.env_response([{"role": "assistant", "content": "hmm let me think"}], state))
        assert not state["done"] and "tool" in obs[0]["content"].lower()
