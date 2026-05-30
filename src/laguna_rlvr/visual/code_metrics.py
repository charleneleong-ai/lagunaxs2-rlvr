"""Safe code-validity metrics for generated output — checks that don't *execute* the code.

The goal-faithful metric is render-diff (execute + render + compare to the target screenshot), but that
needs the code-execution sandbox (report §4.4). These are the cheap, in-process, no-exec companions:
does generated HTML parse, and does generated Python (matplotlib) compile? They catch gibberish without
running anything.
"""
from __future__ import annotations

from html.parser import HTMLParser


def parses_html(text: str) -> bool:
    """Coarse HTML validity: True if `text` has tags and the (lenient) stdlib parser accepts it.

    Rejects plain-text gibberish but NOT malformed HTML (unclosed tags etc.) — a cheap no-exec proxy;
    the strict check is render-diff (sandbox).
    """
    if "<" not in text or ">" not in text:
        return False
    try:
        HTMLParser().feed(text)
        return True
    except Exception:
        return False


def compiles_python(text: str) -> bool:
    """True if `text` is syntactically valid Python (compile only — never executes)."""
    try:
        compile(text, "<generated>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def is_valid(text: str, kind: str | None) -> bool:
    if kind == "html":
        return parses_html(text)
    if kind == "python":
        return compiles_python(text)
    return True  # no code target for this corpus -> not scored


def code_validity_rate(preds: list[str], kinds: list[str | None]) -> float | None:
    """Fraction of code-target predictions that are valid; None if no item has a code target."""
    scored = [(p, k) for p, k in zip(preds, kinds) if k is not None]
    if not scored:
        return None
    return sum(is_valid(p, k) for p, k in scored) / len(scored)
