"""Multi-turn frontend-coding env: read a UI design (via a design tool), write HTML/CSS to match it.

Pairs with the OCR-as-tool path (environments/ocr_tool): the design mockup is 'seen' only through a
tool that returns its spec text (mock OCR / GLM-OCR stand-in), then the agent writes an ```html block.
The env scores the markup against a checklist of structural requirements (regex predicates) — a dense
test-pass-fraction reward, like code_smoke but for markup — so it's a learnable signal with any model
and $0 (no browser, no sandbox). This is the design -> frontend-code half of the visual-context story.
"""
from __future__ import annotations

import random
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
    ("contact.png",
     "Contact form: a name text input, an email input, a message textarea, and a 'Send' button.",
     [("name text input", r"<input[^>]*type=[\"']?text"),
      ("email input", r"<input[^>]*type=[\"']?email"),
      ("message textarea", r"<textarea[\s>]"),
      ("'Send' button", r"<button[^>]*>[^<]*send|type=[\"']?submit")]),
    ("hero.png",
     "Landing hero: a big h1 headline, a tagline paragraph, and a 'Get Started' call-to-action button "
     "on a colored background.",
     [("h1 headline", r"<h1[^>]*>\s*\w"),
      ("tagline paragraph", r"<p[^>]*>\s*\w"),
      ("'Get Started' CTA", r"(<button|<a)[^>]*>[^<]*get started"),
      ("background color", r"background[^;\"']*:[^;\"']*(#[0-9a-f]{3,6}|rgb|blue|teal|indigo|green|purple)")]),
    ("table.png",
     "A users table with a header row (columns Name, Email, Role) and at least one data row.",
     [("table element", r"<table[\s>]"),
      ("header cells", r"<th[\s>]"),
      ("'Name' column", r"<th[^>]*>[^<]*name"),
      ("'Email' column", r"<th[^>]*>[^<]*email"),
      ("data cell", r"<td[\s>]")]),
    ("signup.png",
     "Sign-up card: a 'Create Account' heading, email + password inputs, a terms checkbox, "
     "and a 'Create Account' button.",
     [("heading", r"<h[12][^>]*>[^<]*create account"),
      ("email input", r"<input[^>]*type=[\"']?email"),
      ("password input", r"<input[^>]*type=[\"']?password"),
      ("terms checkbox", r"<input[^>]*type=[\"']?checkbox"),
      ("'Create Account' button", r"<button[^>]*>[^<]*create account")]),
]

_DESIGN_RE = re.compile(r"DESIGN:\s*(\S+)", re.I)

# Synthetic-mockup vocabulary. Each primitive picks values from these pools so every generated
# design gets a distinct multi-requirement checklist (partial credit -> reward variance).
_HEADINGS = ["Dashboard", "Welcome Aboard", "Your Profile", "Order Summary", "Settings",
             "Latest News", "Project Overview", "Account Details", "Get In Touch", "Daily Report"]
_BUTTONS = ["Save", "Continue", "Submit", "Download", "Confirm", "Add Item", "Log Out", "Apply", "Next", "Refresh"]
_LINK_TEXTS = ["Learn More", "View Docs", "Read Blog", "See Pricing", "Browse Gallery", "Open Help"]
_URLS = ["https://example.com/docs", "https://acme.io/pricing", "https://site.dev/blog",
         "https://app.example.com/help", "https://example.org/gallery"]
_ALTS = ["company logo", "product photo", "user avatar", "hero banner", "chart preview"]
_PLACEHOLDERS = ["Enter your name", "Search…", "Your message", "Email address", "Phone number"]
_COLORS = ["#1e90ff", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "teal", "indigo"]
_PARAS = ["Manage everything from one place.", "Built for teams who move fast.",
          "Simple, transparent and reliable.", "Track your progress over time."]


def _esc(text: str) -> str:
    return re.escape(text.lower())


# Each builder: rng -> (requirement_label, regex_pattern, satisfying_html_fragment).
def _heading(rng: random.Random) -> tuple[str, str, str]:
    txt = rng.choice(_HEADINGS)
    return (f"a top heading reading '{txt}'", rf"<h1[^>]*>\s*{_esc(txt)}", f"<h1>{txt}</h1>")


def _button(rng: random.Random) -> tuple[str, str, str]:
    txt = rng.choice(_BUTTONS)
    return (f"a button labeled '{txt}'", rf"<button[^>]*>\s*{_esc(txt)}", f"<button>{txt}</button>")


def _list_items(rng: random.Random) -> tuple[str, str, str]:
    k = rng.randint(2, 5)
    items = "".join("<li>row</li>" for _ in range(k))
    return (f"a list of exactly {k} items", rf"(?:<li[^>]*>.*?</li>\s*){{{k}}}", f"<ul>{items}</ul>")


def _link(rng: random.Random) -> tuple[str, str, str]:
    txt, url = rng.choice(_LINK_TEXTS), rng.choice(_URLS)
    return (f"a '{txt}' link to {url}",
            rf"<a[^>]*href=[\"']?{_esc(url)}[^>]*>[^<]*{_esc(txt)}",
            f'<a href="{url}">{txt}</a>')


def _image(rng: random.Random) -> tuple[str, str, str]:
    alt = rng.choice(_ALTS)
    return (f"an image with alt text '{alt}'", rf"<img[^>]*alt=[\"']?{_esc(alt)}",
            f'<img src="x.png" alt="{alt}">')


def _input(rng: random.Random) -> tuple[str, str, str]:
    ph = rng.choice(_PLACEHOLDERS)
    return (f"an input with placeholder '{ph}'", rf"<input[^>]*placeholder=[\"']?{_esc(ph)}",
            f'<input placeholder="{ph}">')


def _section(rng: random.Random) -> tuple[str, str, str]:
    color = rng.choice(_COLORS)
    return (f"a section with {color} background", rf"background[^;\"']*:[^;\"']*{_esc(color)}",
            f'<section style="background: {color}">block</section>')


def _paragraph(rng: random.Random) -> tuple[str, str, str]:
    txt = rng.choice(_PARAS)
    return (f"a paragraph reading '{txt}'", rf"<p[^>]*>\s*{_esc(txt)}", f"<p>{txt}</p>")


_PRIMITIVES = [_heading, _button, _list_items, _link, _image, _input, _section, _paragraph]


def _synth_designs(n: int, seed: int = 0, *, with_html: bool = False) -> list[tuple]:
    """Deterministically generate n mockups in the env's (mockup_id, design_spec, [(label, regex)]) shape.

    With with_html=True, each design gains a trailing element: a single HTML document that satisfies every
    one of its requirements (a satisfiability witness for the regex checklist) — used only by tests.
    """
    rng = random.Random(seed)
    designs = []
    for i in range(n):
        builders = rng.sample(_PRIMITIVES, rng.randint(3, 5))
        reqs = [b(rng) for b in builders]
        labels = [r[0] for r in reqs]
        spec = "Mockup: " + "; ".join(labels) + "."
        design = (f"design_{i}", spec, [(label, pat) for label, pat, _ in reqs])
        if with_html:
            design += ("<!doctype html><html><body>" + "".join(html for _, _, html in reqs) + "</body></html>",)
        designs.append(design)
    return designs


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
        ds = Dataset.from_list(rows)
        # `dataset` (not just `eval_dataset`) — hosted RL's buffer calls env.get_dataset() on `dataset`.
        super().__init__(dataset=ds, eval_dataset=ds,
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


def load_environment(n_designs: int = 32, seed: int = 0, max_turns: int = 5,
                     efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    # Curated mockups seed the pool; synthetic ones pad it so hosted RL has enough groups to fill the
    # buffer (7 designs starved batch_size=64 / rollouts_per_example=8).
    designs = _BUILTIN_DESIGNS + _synth_designs(n_designs, seed)
    return FrontendDesignEnv(designs, max_turns=max_turns, efficiency_weight=efficiency_weight)
