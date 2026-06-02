"""Tool-call scaffolds — render tool instructions and parse a tool call in several surface syntaxes,
so an env can train/eval a policy ACROSS harnesses instead of overfitting one tool format.

This is MiniMax-M2's multi-scaffold trick ("sample under multiple scaffolds so the policy generalizes
beyond any single tool layout"): the same logical call — ocr(image_id="invoice.png") — is presented as
a line protocol, a Hermes/XML tag, or a JSON object, chosen per task. A policy trained over a mix is
robust to harness change instead of brittle to one tool syntax.

Scope: single-string-arg tool calls (the tool-mediated family — ocr_tool, frontend_design). Multi-arg
code-block tools (general_agent's executed Python) are a different paradigm and an explicit non-goal.

The four text FORMATS vary the *content* syntax (no inference-call change). The `native` path is
different: it advertises tool schemas to the sampler (`tool_defs=to_tool_defs(...)`) so the model emits
structured `tool_calls`, then reads them with `parse_native` — Laguna's real `poolside_v1` deployment
format. `native` needs a tools-capable sampler (vLLM `--tool-call-parser poolside_v1` / Harbor) to
exercise end-to-end, so it's a whole-env mode rather than a per-row text format.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

FORMATS = ("line", "xml", "json", "poolside")


@dataclass(frozen=True)
class Tool:
    name: str
    arg: str           # single primary argument (e.g. "image_id", "value")
    description: str


def resolve_format(index: int, scaffold: str) -> str:
    """A fixed format, or 'mixed' -> round-robin across FORMATS so a batch spans every syntax.
    'native' (structured tool_calls) passes through — it's an env-level mode, not round-robined."""
    if scaffold == "mixed":
        return FORMATS[index % len(FORMATS)]
    if scaffold != "native" and scaffold not in FORMATS:
        raise ValueError(f"unknown scaffold {scaffold!r} (use {FORMATS}, 'native', or 'mixed')")
    return scaffold


def render_instructions(tools: list[Tool], fmt: str) -> str:
    """Describe how to call `tools` in `fmt` — the per-format syntax the policy must produce."""
    catalog = "Available tools:\n" + "\n".join(f"  {t.name}({t.arg}) — {t.description}" for t in tools)
    ex = tools[0]
    if fmt == "line":
        return f"{catalog}\n\nReply with EXACTLY ONE line `<tool>: <{ex.arg}>` (e.g. `{ex.name}: ...`)."
    if fmt == "xml":
        return (f"{catalog}\n\nReply with ONE tool call in tags:\n"
                f'<tool_call>{{"name": "{ex.name}", "arguments": {{"{ex.arg}": "..."}}}}</tool_call>')
    if fmt == "json":
        return f'{catalog}\n\nReply with ONE JSON object:\n{{"tool": "{ex.name}", "{ex.arg}": "..."}}'
    if fmt == "poolside":  # Laguna's native poolside_v1 tool-call dialect
        return (f"{catalog}\n\nReply with ONE tool call:\n"
                f"<tool_call>{ex.name}\n<arg_key>{ex.arg}</arg_key>\n<arg_value>...</arg_value>\n</tool_call>")
    raise ValueError(f"unknown scaffold format {fmt!r}")


_LINE_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+?)\s*$", re.M)
_XML_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S | re.I)
_OBJ_RE = re.compile(r"\{[^{}]*\}", re.S)
# poolside_v1: <tool_call>NAME <arg_key>K</arg_key> <arg_value>V</arg_value> </tool_call>
_POOLSIDE_RE = re.compile(r"<tool_call>\s*(\w+).*?<arg_value>\s*(.*?)\s*</arg_value>", re.S | re.I)


def _clean(value: str) -> str:
    return value.strip().strip(".,'\"")


def parse_call(text: str, fmt: str, tools: list[Tool]) -> tuple[str, str] | None:
    """Return (tool_name, arg_value) for the call the model emitted in `fmt`, or None if none is valid."""
    names = {t.name.lower(): t for t in tools}
    if fmt == "line":
        for m in _LINE_RE.finditer(text):
            if (tool := names.get(m.group(1).lower())):
                return tool.name, _clean(m.group(2))
        return None
    if fmt == "xml":
        m = _XML_RE.search(text)
        return _from_obj(_loads(m.group(1)), names) if m else None
    if fmt == "json":
        for m in _OBJ_RE.finditer(text):
            if (call := _from_obj(_loads(m.group(0)), names)):
                return call
        return None
    if fmt == "poolside":
        m = _POOLSIDE_RE.search(text)
        if not m or (tool := names.get(m.group(1).lower())) is None:
            return None
        return tool.name, _clean(m.group(2))
    raise ValueError(f"unknown scaffold format {fmt!r}")


def _loads(s: str) -> dict | None:
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _from_obj(obj: dict | None, names: dict[str, Tool]) -> tuple[str, str] | None:
    if not obj:
        return None
    if not (tool := names.get(str(obj.get("name") or obj.get("tool") or "").lower())):
        return None
    args = obj["arguments"] if isinstance(obj.get("arguments"), dict) else obj
    val = args.get(tool.arg)
    if val is None:  # tolerate the value under a differently-named key
        extras = [v for k, v in args.items() if k not in ("name", "tool")]
        val = extras[0] if extras else None
    return (tool.name, _clean(str(val))) if val is not None else None


# --- native: structured tool_calls (Laguna's real poolside_v1 deployment path) -------------------

def to_tool_defs(tools: list[Tool]) -> list[dict]:
    """vf.Tool-format schemas to advertise to the sampler (env passes `tool_defs=...`) so the model
    emits structured tool_calls instead of text. (verifiers rejects the legacy OpenAI function wrapper.)"""
    return [{"name": t.name, "description": t.description,
             "parameters": {"type": "object", "properties": {t.arg: {"type": "string"}}, "required": [t.arg]}}
            for t in tools]


def _attr(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def parse_native(message, tools: list[Tool]) -> tuple[str, str] | None:
    """Read a structured tool call off `message.tool_calls`, or None.

    Tolerates both shapes seen in the wild: verifiers' flat JSON-string call
    (`{"name": ..., "arguments": "{...}"}`) and the nested OpenAI form (`{"function": {"name", "arguments"}}`).
    """
    names = {t.name.lower(): t for t in tools}
    for call in _attr(message, "tool_calls") or []:
        if isinstance(call, str):
            call = _loads(call) or {}
        fn = _attr(call, "function") or call   # nested OpenAI, or flat (name/arguments at top level)
        if not (tool := names.get(str(_attr(fn, "name") or "").lower())):
            continue
        raw = _attr(fn, "arguments")
        args = _loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else None)
        if args and (val := args.get(tool.arg)) is not None:
            return tool.name, _clean(str(val))
    return None
