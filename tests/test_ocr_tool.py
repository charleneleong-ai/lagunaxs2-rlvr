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


class TestSynthDocs:
    def test_count_and_hidden_answer_invariant(self):
        docs = ocr_tool._synth_docs(32)
        assert len(docs) == 32
        for iid, text, question, answer in docs:
            assert ocr_tool._norm(answer) in ocr_tool._norm(text), "answer must be OCR-readable"
            # The answer value must never leak into the question — only the field LABEL does.
            assert ocr_tool._norm(answer) not in ocr_tool._norm(question)
            assert iid in question

    def test_distinct_image_ids(self):
        docs = ocr_tool._synth_docs(32)
        assert len({iid for iid, *_ in docs}) == 32

    def test_deterministic(self):
        assert ocr_tool._synth_docs(32, seed=7) == ocr_tool._synth_docs(32, seed=7)


class TestEnv:
    def test_rows_hide_answer_behind_image(self):
        env = ocr_tool.load_environment()
        rows = env.eval_dataset.to_list()
        assert len(rows) >= 24
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


class TestReadScore:
    """Graded read credit: exact -> 1.0, partial overlap in (0, 0.5], disjoint -> 0.0 (no binary cliff)."""

    @pytest.mark.parametrize("gold, guess, lo, hi", [
        ("jane@clinic.org", "jane@clinic.org", 1.0, 1.0),     # exact match
        ("MX-88231", "the id is MX-88231", 1.0, 1.0),         # substring -> match
        ("jane@clinic.org", "jane@clinic.com", 0.01, 0.5),    # near miss -> partial, below exact
        ("MX-88231", "ZZ-00000", 0.0, 0.0),                   # disjoint -> 0
    ])
    def test_graded(self, gold, guess, lo, hi):
        assert lo <= ocr_tool._read_score(gold, guess) <= hi

    def test_partial_answer_gets_nonzero_reward(self):
        env = ocr_tool.load_environment()
        state = {"info": {"image_id": "c.png", "text": "Email: jane@clinic.org", "answer": "jane@clinic.org"},
                 "turn": 1}
        asyncio.run(env.setup_state(state))
        asyncio.run(env.env_response([{"role": "assistant", "content": "ANSWER: jane@clinic.com"}], state))
        assert not state["solved"] and 0.0 < env._reward(state) < 1.0  # graded, not a 0 cliff


def test_get_dataset_is_set_for_training():
    """Hosted RL's buffer does exactly this — env.get_dataset(seed=...). It must NOT raise
    'dataset is not set' (the ValueError that killed the Prime run at buffer init: training reads
    `dataset`, not `eval_dataset`). This local check catches that class of failure before any push."""
    ds = ocr_tool.load_environment().get_dataset(seed=0)
    assert ds is not None and len(ds) >= 24
