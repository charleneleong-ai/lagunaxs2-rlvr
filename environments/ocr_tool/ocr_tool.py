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

import random
import re

import verifiers as vf
from datasets import Dataset

from laguna_rlvr.code_exec import message_text   # vendor before any Hub push
from laguna_rlvr.rewards import RolloutState, binary, efficiency_bonus

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

# Synthetic doc templates: each is (image_kind, title, [(label, value-generator)]). One field per doc is
# picked as the target so the question names only the LABEL — the value lives solely in the doc text, so
# answering requires an OCR call (mirrors the hidden-answer invariant of _BUILTIN_DOCS).
_FIRST_NAMES = ["Alex", "Jordan", "Sam", "Riley", "Casey", "Morgan", "Taylor", "Jamie", "Drew", "Quinn"]
_LAST_NAMES = ["Rivera", "Chen", "Patel", "Nguyen", "Okafor", "Silva", "Kim", "Haddad", "Novak", "Reyes"]
_DOMAINS = ["clinic.org", "acme.com", "mail.net", "studio.io", "ledger.co"]


def _name(r: random.Random) -> str:
    return f"{r.choice(_FIRST_NAMES)} {r.choice(_LAST_NAMES)}"


def _money(r: random.Random) -> str:
    return f"${r.randint(10, 999)}.{r.randint(0, 99):02d}"


def _date(r: random.Random) -> str:
    return f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"


def _phone(r: random.Random) -> str:
    return f"555-{r.randint(100, 999):03d}{r.randint(0, 9)}"


_TEMPLATES = [
    ("invoice", "ACME Supplies — Invoice", [
        ("Invoice Number", lambda r: f"INV-{r.randint(1000, 9999)}"),
        ("Date", _date),
        ("Total Due", _money),
        ("Account", lambda r: f"AC-{r.randint(1000, 9999)}"),
    ]),
    ("receipt", "Blue Bottle Coffee — Receipt", [
        ("Order Number", lambda r: f"{r.randint(100, 999)}"),
        ("Total", _money),
        ("Card Last Four", lambda r: f"{r.randint(0, 9999):04d}"),
        ("Date", _date),
    ]),
    ("card", "Business Card", [
        ("Name", _name),
        ("Phone", _phone),
        ("Email", lambda r: f"{r.choice(_FIRST_NAMES).lower()}@{r.choice(_DOMAINS)}"),
        ("Title", lambda r: r.choice(["Cardiologist", "Architect", "Engineer", "Designer", "Analyst"])),
    ]),
    ("form", "Membership Form", [
        ("Name", _name),
        ("Date of Birth", lambda r: f"19{r.randint(60, 99)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"),
        ("Member ID", lambda r: f"MX-{r.randint(10000, 99999)}"),
        ("Plan", lambda r: r.choice(["Gold", "Silver", "Bronze", "Platinum"])),
    ]),
]


def _synth_docs(n: int, seed: int = 0) -> list[tuple[str, str, str, str]]:
    """Deterministically generate `n` (image_id, document_text, question, answer) docs. Each doc fills
    every field of one template; one field is the target — the question names only its LABEL while the
    value lives solely in document_text, so the answer is readable only via OCR (hidden-answer invariant)."""
    r = random.Random(seed)
    docs = []
    for i in range(n):
        kind, title, fields = _TEMPLATES[i % len(_TEMPLATES)]
        values = [(label, gen(r)) for label, gen in fields]
        target_label, answer = values[r.randrange(len(values))]
        image_id = f"doc_{i}.png"
        text = "\n".join([title] + [f"{label}: {val}" for label, val in values])
        question = f"What is the {target_label} on {image_id}? Reply with just the value."
        docs.append((image_id, text, question, answer))
    return docs


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


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _read_score(gold: str, guess: str) -> float:
    """Graded read credit: 1.0 on a match (exact / float / substring), else token-F1 partial credit
    (x0.5, always below an exact read). A near-miss read still earns gradient, so groups aren't all-zero
    at cold-start — the binary-reward trap that needs G to be large to escape (the local GSPO lesson)."""
    if _match(gold, guess):
        return 1.0
    g, h = _words(gold), _words(guess)
    return 0.5 * (2 * len(g & h) / (len(g) + len(h))) if g and h else 0.0


def ocr(image_id: str, images: dict[str, str]) -> str:
    """Mock OCR backend: return the image's known text (perfect extraction; stands in for GLM-OCR).
    A real backend (GLM-OCR, tesseract) keeps the same (image_id, store) seam, rendering/extracting instead."""
    return images.get(image_id, f"(no image named '{image_id}')")


class OCRToolEnv(vf.MultiTurnEnv):
    def __init__(self, docs, *, max_turns: int, efficiency_weight: float, **kwargs):
        self._eff_w = efficiency_weight
        rows = [{"question": f"{_PROTOCOL}\n\nTask: {q}", "answer": a,
                 "info": {"image_id": iid, "text": txt, "answer": a}} for iid, txt, q, a in docs]
        ds = Dataset.from_list(rows)
        # `dataset` (not just `eval_dataset`) — hosted RL's buffer calls env.get_dataset(), which reads
        # `dataset` and raises "dataset is not set" otherwise (the run that died at buffer init).
        super().__init__(dataset=ds, eval_dataset=ds,
                         rubric=vf.Rubric(funcs=[self._reward, self._success], weights=[1.0, 0.0]),
                         max_turns=max_turns, message_type="chat", **kwargs)

    def _rs(self, state) -> RolloutState:
        solved = bool(state.get("solved", False))
        return RolloutState(tests_passed=int(solved), tests_total=1, turns=int(state.get("turn", 0)),
                            max_turns=self.max_turns, succeeded=solved)

    def _reward(self, state, **_) -> float:
        # graded read credit (exact 1.0 / partial token-overlap) + an efficiency nudge paid only on a
        # full solve — dense enough that cold-start groups have reward variance, not all-zeros.
        return state.get("read_score", 0.0) + self._eff_w * efficiency_bonus(self._rs(state))

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
            state["read_score"] = _read_score(info["answer"], ans.group(1))  # graded, not just binary
            state["done"] = True
            return [{"role": "user", "content": "✅ Correct." if state["solved"] else "❌ Incorrect."}]
        if (call := _OCR_RE.search(text)):
            image_id = call.group(1).strip().strip(".,'\"")
            extracted = ocr(image_id, {info["image_id"]: info["text"]})
            return [{"role": "user", "content": f"[OCR of {image_id}]\n{extracted}\n\n"
                     "Now reply 'OCR: <image_id>' or 'ANSWER: <value>'."}]
        return [{"role": "user", "content": "Reply with exactly one line: "
                 "'OCR: <image_id>' or 'ANSWER: <value>'."}]


def load_environment(n_docs: int = 32, seed: int = 0, max_turns: int = 4,
                     efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    # Curated hand-written docs seed the pool; synthetic docs pad it out so a hosted RL run has enough
    # zero-advantage-survivable groups (4 docs starved batch_size=64 / rollouts_per_example=8).
    docs = _BUILTIN_DOCS + _synth_docs(n_docs, seed)
    return OCRToolEnv(docs, max_turns=max_turns, efficiency_weight=efficiency_weight)
