import asyncio
import importlib.util
import sys
from pathlib import Path

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
    def test_designs_have_requirements_and_hide_spec(self):
        env = fd.load_environment()
        rows = env.eval_dataset.to_list()
        assert len(rows) >= len(fd._BUILTIN_DESIGNS)  # curated seed + synthetic pool
        for row in rows:
            assert row["info"]["labels"] and len(row["info"]["labels"]) == len(row["info"]["patterns"])
            # the spec text is behind the DESIGN tool, not leaked into the prompt
            assert row["info"]["spec"] not in row["question"]

    def test_partial_credit_then_full_solve(self):
        env = fd.load_environment()
        info = {"mockup_id": "login.png", "spec": "...",
                "labels": _LABELS, "patterns": _PATTERNS}
        state = {"info": info, "turn": 1}
        asyncio.run(env.setup_state(state))
        # partial: email only -> 1/2, not solved
        asyncio.run(env.env_response([{"role": "assistant", "content": '```html\n<input type="email">\n```'}], state))
        assert state["passed"] == 1 and not state["solved"]
        # full: both -> solved
        asyncio.run(env.env_response(
            [{"role": "assistant", "content": '```html\n<input type="email"><button>Submit</button>\n```'}], state))
        assert state["passed"] == 2 and state["solved"]

    def test_design_tool_returns_spec(self):
        env = fd.load_environment()
        state = {"info": {"mockup_id": "login.png", "spec": "a blue Sign In button",
                          "labels": _LABELS, "patterns": _PATTERNS}, "turn": 1}
        asyncio.run(env.setup_state(state))
        obs = asyncio.run(env.env_response([{"role": "assistant", "content": "DESIGN: login.png"}], state))
        assert "blue Sign In button" in obs[0]["content"] and not state["solved"]


def test_get_dataset_is_set_for_training():
    """Hosted RL's buffer calls env.get_dataset(seed=...) on `dataset` — must not raise 'dataset is
    not set'. Local guard for the failure that killed the Prime run at buffer init."""
    ds = fd.load_environment().get_dataset(seed=0)
    assert ds is not None and len(ds) >= 24
