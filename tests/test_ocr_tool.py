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


class TestEnv:
    def test_builtin_rows_hide_answer_behind_image(self):
        env = ocr_tool.load_environment()
        rows = env.eval_dataset.to_list()
        assert len(rows) == len(ocr_tool._BUILTIN_DOCS)
        for row in rows:
            # The answer must NOT be readable from the prompt — it lives only in the OCR-able doc.
            assert ocr_tool._norm(row["info"]["answer"]) not in ocr_tool._norm(row["question"])
            assert row["info"]["text"], "doc text needed for the ocr tool"

    def test_ocr_then_correct_answer_solves(self):
        env = ocr_tool.load_environment()
        state = {"info": {"image_id": "invoice.png", "text": "Total Due: $42.50", "answer": "42.50"},
                 "turn": 1}
        asyncio.run(env.setup_state(state))
        obs = asyncio.run(env.env_response([{"role": "assistant", "content": "OCR: invoice.png"}], state))
        assert "42.50" in obs[0]["content"] and not state["done"]
        asyncio.run(env.env_response([{"role": "assistant", "content": "ANSWER: 42.50"}], state))
        assert state["solved"] and state["done"]

    def test_wrong_answer_is_done_but_unsolved(self):
        env = ocr_tool.load_environment()
        state = {"info": {"image_id": "x.png", "text": "Total: $9", "answer": "9"}, "turn": 1}
        asyncio.run(env.setup_state(state))
        asyncio.run(env.env_response([{"role": "assistant", "content": "ANSWER: 42"}], state))
        assert state["done"] and not state["solved"]
