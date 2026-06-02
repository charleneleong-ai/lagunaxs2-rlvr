"""Multi-turn OCR-as-tool eval: the model can't see the document image — it must call ocr(image_id)
to read the text, then answer a question about a specific field.

This is the *tool-mediated* path to visual context (the cheap alternative to a trained projector
adapter — see docs/a100-multimodal-adapter.md): an OCR encoder bridges image -> text and a text LLM
reasons over the result, so it runs on ANY OpenAI-compatible model with no vision weights and $0
(Ollama / Prime inference).

The OCR backend is pluggable. The default 'mock' backend returns each document's known text — perfect
extraction, standing in for GLM-OCR; swap in a real render+OCR backend to also measure extraction
noise. Either way the eval scores the agent loop: decide to call ocr, parse the right field, answer.
"""
from __future__ import annotations

import re

import verifiers as vf
from datasets import Dataset

from laguna_rlvr.code_exec import message_text   # vendor before any Hub push
from laguna_rlvr.rewards import RolloutState, binary, shaped
from laguna_rlvr.scaffold import (Tool, parse_call, parse_native, render_instructions,
                                  resolve_format, to_tool_defs)

# (image_id, document_text, question, answer) — multi-field docs so answering needs parsing, not echo.
_BUILTIN_DOCS = [
    ("invoice.png",
     "ACME Supplies\nInvoice #1042\nDate: 2026-05-01\nSubtotal: $38.00\nTax: $4.50\nTotal Due: $42.50",
     "What is the total amount due on invoice.png? Reply with just the number.", "42.50"),
    ("receipt.png",
     "Blue Bottle Coffee\nLatte         5.50\nCroissant     4.00\nTotal         9.50\nVISA ****1234",
     "What are the last four digits of the card used on receipt.png?", "1234"),
    ("card.png",
     "Dr. Jane Smith\nCardiologist\nPhone: 555-0142\nEmail: jane@clinic.org",
     "What is the email address on card.png?", "jane@clinic.org"),
    ("form.png",
     "Membership Form\nName: Alex Rivera\nDOB: 1990-03-14\nMember ID: MX-88231\nPlan: Gold",
     "What is the Member ID on form.png?", "MX-88231"),
]

# OCR-as-tool + answer modeled as tools, so the scaffold can render/parse the call in any syntax.
_TOOLS = [Tool("ocr", "image_id", "returns the text an OCR encoder extracts from that image"),
          Tool("answer", "value", "submit your final answer — just the value")]

_NORM_RE = re.compile(r"[\s$,]")


def _norm(s: str) -> str:
    return _NORM_RE.sub("", s.strip().lower())


def _match(gold: str, guess: str) -> bool:
    """Lenient answer check: normalized equality, float-equality, or gold appearing in the guess."""
    g, h = _norm(gold), _norm(guess)
    if not g:
        return False
    if g == h:
        return True
    try:
        return abs(float(g) - float(h)) < 1e-6
    except ValueError:
        return g in h   # tolerate surrounding words ("the email is jane@clinic.org")


def ocr(image_id: str, images: dict[str, str]) -> str:
    """Mock OCR backend: return the image's known text (perfect extraction; stands in for GLM-OCR).
    A real backend (GLM-OCR, tesseract) keeps the same (image_id, store) seam, rendering/extracting instead."""
    return images.get(image_id, f"(no image named '{image_id}')")


class OCRToolEnv(vf.MultiTurnEnv):
    def __init__(self, docs, *, max_turns: int, efficiency_weight: float, scaffold: str, **kwargs):
        self._eff_w = efficiency_weight
        rows = []
        for i, (iid, txt, q, a) in enumerate(docs):
            fmt = resolve_format(i, scaffold)
            instr = ("Use the available tools to read the image, then answer." if fmt == "native"
                     else render_instructions(_TOOLS, fmt))
            question = f"You have a task about a document IMAGE you cannot see directly.\n\nTask: {q}\n\n{instr}"
            rows.append({"question": question, "answer": a,
                         "info": {"image_id": iid, "text": txt, "answer": a, "fmt": fmt}})
        if scaffold == "native":   # advertise structured tool schemas to the sampler
            kwargs.setdefault("tool_defs", to_tool_defs(_TOOLS))
        super().__init__(eval_dataset=Dataset.from_list(rows),
                         rubric=vf.Rubric(funcs=[self._reward, self._success], weights=[1.0, 0.0]),
                         max_turns=max_turns, message_type="chat", **kwargs)

    def _rs(self, state) -> RolloutState:
        solved = bool(state.get("solved", False))
        return RolloutState(tests_passed=int(solved), tests_total=1, turns=int(state.get("turn", 0)),
                            max_turns=self.max_turns, succeeded=solved)

    def _reward(self, state, **_) -> float:
        return shaped(self._rs(state), self._eff_w)

    def _success(self, state, **_) -> float:
        return binary(self._rs(state))

    async def setup_state(self, state) -> None:
        state["solved"] = False
        # `done` (not just `solved`) is the stop flag: a wrong terminal ANSWER ends the rollout too,
        # which `solved` alone can't express — hence one more flag than the sibling general_agent env.
        state["done"] = False

    @vf.stop
    async def is_done(self, state) -> bool:
        return bool(state.get("done"))

    async def env_response(self, messages, state, **kwargs):
        info, last = state["info"], messages[-1]
        call = (parse_native(last, _TOOLS) if info["fmt"] == "native"
                else parse_call(message_text(last), info["fmt"], _TOOLS))
        if call is None:
            hint = ("Call one of the available tools." if info["fmt"] == "native"
                    else render_instructions(_TOOLS, info["fmt"]))
            return [{"role": "user", "content": "No valid tool call found.\n" + hint}]
        name, value = call
        if name == "answer":
            state["solved"] = _match(info["answer"], value)
            state["done"] = True
            return [{"role": "user", "content": "✅ Correct." if state["solved"] else "❌ Incorrect."}]
        extracted = ocr(value, {info["image_id"]: info["text"]})
        return [{"role": "user", "content": f"[ocr of {value}]\n{extracted}\n\nNow call a tool."}]


def load_environment(max_turns: int = 4, efficiency_weight: float = 0.1,
                     scaffold: str = "mixed", **kwargs) -> vf.Environment:
    """scaffold: 'line'|'xml'|'json'|'poolside' (text syntax) · 'mixed' (round-robin) · 'native' (structured tool_calls)."""
    return OCRToolEnv(_BUILTIN_DOCS, max_turns=max_turns, efficiency_weight=efficiency_weight,
                      scaffold=scaffold)
