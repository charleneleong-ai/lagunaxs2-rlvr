"""Multi-turn frontend-coding env: read a UI design (via a design tool), write HTML/CSS to match it.

Pairs with the OCR-as-tool path (environments/ocr_tool): the design mockup is 'seen' only through a
tool that returns its spec text (mock OCR / GLM-OCR stand-in), then the agent writes an ```html block.
The env scores the markup against a checklist of structural requirements (regex predicates) — a dense
test-pass-fraction reward, like code_smoke but for markup — so it's a learnable signal with any model
and $0 (no browser, no sandbox). This is the design -> frontend-code half of the visual-context story.
"""
from __future__ import annotations

import re

import verifiers as vf
from datasets import Dataset

from laguna_rlvr.code_exec import extract_code, message_text   # vendor before any Hub push
from laguna_rlvr.rewards import RolloutState, binary, shaped

# (mockup_id, design_spec, [(requirement_label, regex)]) — regexes checked case-insensitively on the markup.
_BUILTIN_DESIGNS = [
    ("login.png",
     "Login screen: a 'Welcome Back' heading (h1), an email input, a password input, "
     "and a blue 'Sign In' button.",
     [("'Welcome Back' h1", r"<h1[^>]*>\s*welcome back"),
      ("email input", r"<input[^>]*type=[\"']?email"),
      ("password input", r"<input[^>]*type=[\"']?password"),
      ("'Sign In' button", r"<button[^>]*>[^<]*sign in"),
      ("blue styling", r"#[0-9a-f]{3,6}|blue|rgb\(")]),
    ("pricing.png",
     "Pricing card: a 'Pro' title, the price '$29/mo', a bulleted list of features, "
     "and a 'Subscribe' button.",
     [("'Pro' title", r">\s*pro\b"),
      ("$29 price", r"\$29"),
      ("feature list", r"<ul[\s>]|<ol[\s>]"),
      ("'Subscribe' button", r"<button[^>]*>[^<]*subscribe")]),
    ("navbar.png",
     "Top navigation bar (nav): brand 'Acme' on the left and three links — Home, About, Contact.",
     [("nav element", r"<nav[\s>]"),
      ("'Acme' brand", r">\s*acme\b"),
      ("Home link", r"<a[^>]*>[^<]*home"),
      ("About link", r"<a[^>]*>[^<]*about"),
      ("Contact link", r"<a[^>]*>[^<]*contact")]),
]

_DESIGN_RE = re.compile(r"DESIGN:\s*(\S+)", re.I)


def score_markup(html: str, labels: list[str], patterns: list[str]) -> list[str]:
    """Return the labels of requirements NOT satisfied by the markup (case-insensitive regex search)."""
    return [label for label, pat in zip(labels, patterns) if not re.search(pat, html, re.I | re.S)]


def _prompt(mockup_id: str) -> str:
    return (f"You are a frontend engineer. Implement the UI design in '{mockup_id}' as a single HTML "
            f"document (inline CSS is fine). You cannot see the image directly.\n\n"
            f"Tools, one per line:\n  DESIGN: {mockup_id}   — returns the design spec the OCR encoder reads\n\n"
            "Read the design first, then reply with an ```html code block. You'll get a list of any unmet "
            "requirements — fix them and resend the block until all pass.")


class FrontendDesignEnv(vf.MultiTurnEnv):
    def __init__(self, designs, *, max_turns: int, efficiency_weight: float, **kwargs):
        self._eff_w = efficiency_weight
        rows = [{"question": _prompt(mid), "answer": "",
                 "info": {"mockup_id": mid, "spec": spec,
                          "labels": [r[0] for r in reqs], "patterns": [r[1] for r in reqs]}}
                for mid, spec, reqs in designs]
        super().__init__(eval_dataset=Dataset.from_list(rows),
                         rubric=vf.Rubric(funcs=[self._reward, self._success], weights=[1.0, 0.0]),
                         max_turns=max_turns, message_type="chat", **kwargs)

    def _rs(self, state) -> RolloutState:
        return RolloutState(tests_passed=int(state.get("passed", 0)), tests_total=int(state.get("total", 1)),
                            turns=int(state.get("turn", 0)), max_turns=self.max_turns,
                            succeeded=bool(state.get("solved", False)))

    def _reward(self, state, **_) -> float:
        return shaped(self._rs(state), self._eff_w)

    def _success(self, state, **_) -> float:
        return binary(self._rs(state))

    async def setup_state(self, state) -> None:
        state["solved"] = False
        state["passed"] = 0
        state["total"] = len(state["info"]["labels"])

    @vf.stop
    async def is_solved(self, state) -> bool:
        return bool(state.get("solved", False))

    async def env_response(self, messages, state, **kwargs):
        text = message_text(messages[-1])
        info = state["info"]
        if "```" not in text and (m := _DESIGN_RE.search(text)):
            mid = m.group(1).strip().strip(".,'\"")
            spec = info["spec"] if mid == info["mockup_id"] else f"(no design named '{mid}')"
            return [{"role": "user", "content": f"[design {mid}]\n{spec}\n\nNow reply with an ```html block."}]
        if "```" in text:
            unmet = score_markup(extract_code(text), info["labels"], info["patterns"])
            state["passed"] = len(info["labels"]) - len(unmet)
            if not unmet:
                state["solved"] = True
                return [{"role": "user", "content": "✅ All requirements met."}]
            return [{"role": "user", "content": "Unmet requirements:\n" + "\n".join(f"- {u}" for u in unmet) +
                     "\nFix and resend the ```html block."}]
        return [{"role": "user", "content": "Read the design with `DESIGN: <mockup_id>`, then send an ```html block."}]


def load_environment(max_turns: int = 5, efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    return FrontendDesignEnv(_BUILTIN_DESIGNS, max_turns=max_turns, efficiency_weight=efficiency_weight)
