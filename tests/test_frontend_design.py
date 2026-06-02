import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from scaffold_emit import emit_call

_PATH = Path(__file__).parent.parent / "environments" / "frontend_design" / "frontend_design.py"
_spec = importlib.util.spec_from_file_location("frontend_design", _PATH)
fd = importlib.util.module_from_spec(_spec)
sys.modules["frontend_design"] = fd
_spec.loader.exec_module(fd)

_LABELS = ["email input", "submit button"]
_PATTERNS = [r"<input[^>]*type=[\"']?email", r"<button[^>]*>[^<]*submit"]


class TestScoreMarkup:
    def test_all_met_returns_empty(self):
        html = '<input type="email"><button>Submit</button>'
        assert fd.score_markup(html, _LABELS, _PATTERNS) == []

    def test_reports_only_unmet(self):
        html = '<input type="email">'  # missing the button
        assert fd.score_markup(html, _LABELS, _PATTERNS) == ["submit button"]

    def test_case_insensitive(self):
        html = '<INPUT TYPE="EMAIL"><BUTTON>SUBMIT</BUTTON>'
        assert fd.score_markup(html, _LABELS, _PATTERNS) == []


class TestEnv:
    def test_builtin_designs_hide_spec_and_mix_scaffolds(self):
        env = fd.load_environment(scaffold="mixed")
        rows = env.eval_dataset.to_list()
        assert len(rows) == len(fd._BUILTIN_DESIGNS)
        for row in rows:
            assert row["info"]["labels"] and len(row["info"]["labels"]) == len(row["info"]["patterns"])
            assert row["info"]["spec"] not in row["question"]   # spec is behind the read_design tool
        assert len({r["info"]["fmt"] for r in rows}) >= 2, "'mixed' must vary the scaffold across rows"

    def test_partial_credit_then_full_solve(self):
        env = fd.load_environment(scaffold="line")
        state = {"info": {"mockup_id": "login.png", "spec": "...", "fmt": "line",
                          "labels": _LABELS, "patterns": _PATTERNS}, "turn": 1}
        asyncio.run(env.setup_state(state))
        # partial: email only -> 1/2, not solved
        asyncio.run(env.env_response([{"role": "assistant", "content": '```html\n<input type="email">\n```'}], state))
        assert state["passed"] == 1 and not state["solved"]
        # full: both -> solved
        asyncio.run(env.env_response(
            [{"role": "assistant", "content": '```html\n<input type="email"><button>Submit</button>\n```'}], state))
        assert state["passed"] == 2 and state["solved"]

    @pytest.mark.parametrize("fmt", ["line", "xml", "json", "poolside"])
    def test_read_design_tool_returns_spec_in_each_scaffold(self, fmt):
        env = fd.load_environment(scaffold=fmt)
        state = {"info": {"mockup_id": "login.png", "spec": "a blue Sign In button", "fmt": fmt,
                          "labels": _LABELS, "patterns": _PATTERNS}, "turn": 1}
        asyncio.run(env.setup_state(state))
        call = emit_call(fmt, "read_design", "mockup_id", "login.png")
        obs = asyncio.run(env.env_response([{"role": "assistant", "content": call}], state))
        assert "blue Sign In button" in obs[0]["content"] and not state["solved"]
