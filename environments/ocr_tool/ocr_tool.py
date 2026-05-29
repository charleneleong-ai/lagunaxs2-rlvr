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

_PROTOCOL = (
    "You are given a task about a document IMAGE you cannot see directly. You have one tool:\n"
    "  OCR: <image_id>   — returns the text an OCR encoder extracts from that image.\n\n"
    "Each turn reply with EXACTLY ONE line, either:\n"
    "  OCR: <image_id>     to read the document, or\n"
    "  ANSWER: <value>     once you know the answer (just the value)."
)


_NORM_RE = re.compile(r"[\s$,]")
_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.I)
_OCR_RE = re.compile(r"OCR:\s*(\S+)", re.I)


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
    def __init__(self, docs, *, max_turns: int, efficiency_weight: float, **kwargs):
        self._eff_w = efficiency_weight
        rows = [{"question": f"{_PROTOCOL}\n\nTask: {q}", "answer": a,
                 "info": {"image_id": iid, "text": txt, "answer": a}} for iid, txt, q, a in docs]
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
        text = message_text(messages[-1])
        info = state["info"]
        if (ans := _ANSWER_RE.search(text)):
            state["solved"] = _match(info["answer"], ans.group(1))
            state["done"] = True
            return [{"role": "user", "content": "✅ Correct." if state["solved"] else "❌ Incorrect."}]
        if (call := _OCR_RE.search(text)):
            image_id = call.group(1).strip().strip(".,'\"")
            extracted = ocr(image_id, {info["image_id"]: info["text"]})
            return [{"role": "user", "content": f"[OCR of {image_id}]\n{extracted}\n\n"
                     "Now reply 'OCR: <image_id>' or 'ANSWER: <value>'."}]
        return [{"role": "user", "content": "Reply with exactly one line: "
                 "'OCR: <image_id>' or 'ANSWER: <value>'."}]


def load_environment(max_turns: int = 4, efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    return OCRToolEnv(_BUILTIN_DOCS, max_turns=max_turns, efficiency_weight=efficiency_weight)
