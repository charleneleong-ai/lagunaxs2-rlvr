"""Vision-grounded code-editing eval — *see the rendered artifact, apply an instruction, output the
edited source*. Language-keyed: the input `kind` (html / python / …) selects the per-language plugin
(a `mutate` that synthesizes a ground-truth edit from an (image, code) pair, and a `validate`). Scored
by **edit-applied** (the requested change is present) + **still-valid** (the output parses). Render-sim
(re-render the edit, compare to the target) is the natural third score — left pluggable (needs a
per-language renderer) and deferred; `EditTask.edited` keeps the ground-truth render target for it.

This is a *capability* eval (the target — vision-grounded editing), not a perception one: it measures
what we're building toward, dispatched on language exactly as a coding model should be evaluated.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

from laguna_rlvr.visual.corpora import CORPUS_KIND
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter


@dataclass
class EditTask:
    instruction: str  # the visually-grounded edit to apply
    check: str        # the value that must appear in the output if the edit was applied
    edited: str       # the ground-truth edited source (kept for render-sim, later)
    kind: str


# ── per-language plugins: mutate (synthesize a ground-truth edit) + validate ───────────────────────

_NEW_TITLES = ["Acme Analytics", "Blue Harbor Cafe", "Quantum Labs", "Riverside Clinic",
               "Nova Robotics", "Green Valley Farm", "Summit Consulting", "Pixel Forge Studio"]
_H1 = re.compile(r"(<h1[^>]*>)(.*?)(</h1>)", re.I | re.S)
_TITLE = re.compile(r"(<title[^>]*>)(.*?)(</title>)", re.I | re.S)
_PY_TITLE = re.compile(r"""((?:set_title|suptitle|plt\.title)\(\s*['"])([^'"]+)(['"])""")


def _html_mutate(code: str, rng: random.Random) -> tuple[str, str, str] | None:
    new = rng.choice(_NEW_TITLES)
    for pat in (_H1, _TITLE):  # prefer the visible <h1>, fall back to <title>
        if pat.search(code):
            return (f'Change the main heading of this page to "{new}".',
                    pat.sub(rf"\g<1>{new}\g<3>", code, count=1), new)
    return None


def _html_validate(code: str) -> bool:
    from html.parser import HTMLParser
    try:
        HTMLParser().feed(code)
    except Exception:
        return False
    return "<" in code and ">" in code


def _py_mutate(code: str, rng: random.Random) -> tuple[str, str, str] | None:
    new = rng.choice(_NEW_TITLES)
    if _PY_TITLE.search(code):
        return (f'Change the chart title to "{new}".', _PY_TITLE.sub(rf"\g<1>{new}\g<3>", code, count=1), new)
    return None


def _py_validate(code: str) -> bool:
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


_PLUGINS: dict[str, tuple] = {  # kind -> (mutate, validate); render-sim renderer added later
    "html": (_html_mutate, _html_validate),
    "python": (_py_mutate, _py_validate),
}


# ── runner ─────────────────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def edit_eval(adapter: VisualAdapter, items: list, *, with_code_context: bool = True,
              max_new_tokens: int = 512, code_chars: int = 1500, seed: int = 0) -> dict[str, float]:
    """`items`: (image, code, corpus). For each, synthesize a language-appropriate edit, ask the adapter
    to apply it (image [+ current code] + instruction), and score edit-applied + still-valid, per kind."""
    rng = random.Random(seed)
    per_kind: dict[str, dict] = defaultdict(lambda: {"applied": 0, "valid": 0, "n": 0})
    for img, code, corpus in items:
        plug = _PLUGINS.get(CORPUS_KIND.get(corpus))
        if not plug:
            continue
        mutate, validate = plug
        if not (m := mutate(code, rng)):
            continue
        instruction, _edited, check = m
        prompt = f"{IMAGE_TOKEN}\n{instruction}"
        if with_code_context:
            prompt += f"\nCurrent code:\n{code[:code_chars]}"
        prompt += "\nReturn the full edited code."
        out = adapter.chat([Turn(prompt, [img])], max_new_tokens=max_new_tokens)[0]
        kind = CORPUS_KIND.get(corpus)
        per_kind[kind]["n"] += 1
        per_kind[kind]["applied"] += int(_norm(check) in _norm(out))  # requested change present
        per_kind[kind]["valid"] += int(validate(out))                 # output still parses
    out: dict[str, float] = {}
    for kind, s in per_kind.items():
        out[f"edit/{kind}/applied"] = s["applied"] / max(s["n"], 1)
        out[f"edit/{kind}/valid"] = s["valid"] / max(s["n"], 1)
    if per_kind:
        agg = {k: sum(s[k] for s in per_kind.values()) for k in ("applied", "valid", "n")}
        out["edit/applied"] = agg["applied"] / max(agg["n"], 1)
        out["edit/valid"] = agg["valid"] / max(agg["n"], 1)
    return out
