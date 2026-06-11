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

import json
import re
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from laguna_rlvr.code_exec import message_text   # vendor before any Hub push
from laguna_rlvr.rewards import RolloutState, binary, shaped
from laguna_rlvr.scaffold import (Tool, parse_call, parse_native, render_instructions,
                                  resolve_format, to_tool_defs)

# Each doc spans the glyph subset where the visual adapter hit the wall (docvqa/ocrvqa/infographic/
# visualmrc/chart-text), tagged by `cat` so the baseline breaks down per glyph-task-type. Multi-field
# with distractors so answering needs parsing the right field, not echoing the only candidate. The
# answer never appears in the question — it lives only in the OCR-able text (enforced by the tests).
_BUILTIN_DOCS = [
    # docvqa — dense business documents / forms: field extraction among distractor values
    dict(cat="docvqa", id="invoice.png",
         text="ACME Supplies\nInvoice #1042\nDate: 2026-05-01\nSubtotal: $38.00\nTax: $4.50\nTotal Due: $42.50",
         q="What is the total amount due on invoice.png? Reply with just the number.", a="42.50"),
    dict(cat="docvqa", id="receipt.png",
         text="Blue Bottle Coffee\nLatte         5.50\nCroissant     4.00\nTotal         9.50\nVISA ****1234",
         q="What are the last four digits of the card used on receipt.png?", a="1234"),
    dict(cat="docvqa", id="card.png",
         text="Dr. Jane Smith\nCardiologist\nPhone: 555-0142\nEmail: jane@clinic.org",
         q="What is the email address on card.png?", a="jane@clinic.org"),
    dict(cat="docvqa", id="form.png",
         text="Membership Form\nName: Alex Rivera\nDOB: 1990-03-14\nMember ID: MX-88231\nPlan: Gold",
         q="What is the Member ID on form.png?", a="MX-88231"),
    dict(cat="docvqa", id="purchase_order.png",
         text="Northwind Traders\nPurchase Order PO-7781\nVendor: Globex\nItem: A4 Paper x 12\n"
              "Item: Toner x 3\nShip By: 2026-06-20\nTotal: $214.00",
         q="How many units of Toner were ordered on purchase_order.png?", a="3"),
    dict(cat="docvqa", id="bank_statement.png",
         text="First National Bank\nStatement Period: 2026-04\nOpening Balance: $1,240.00\n"
              "Deposits: $800.00\nWithdrawals: $310.00\nClosing Balance: $1,730.00",
         q="What is the closing balance on bank_statement.png?", a="1730.00"),
    # ocrvqa — book covers: title/author/publisher/year, answer is one named field
    dict(cat="ocrvqa", id="novel_cover.png",
         text="THE GLASS ATLAS\na novel by Mara Halloway\nPenguin Classics\nThird Edition, 2019",
         q="Who is the author of the book on novel_cover.png?", a="Mara Halloway"),
    dict(cat="ocrvqa", id="cookbook_cover.png",
         text="SAVORY ROOTS\nVegetarian Cooking for Every Season\nby Tomas Lindqvist\nHarvest Press · 2021",
         q="In what year was the book on cookbook_cover.png published?", a="2021"),
    # infographic — stat callouts: pick a percentage / compare regions
    dict(cat="infographic", id="survey_infographic.png",
         text="REMOTE WORK 2026\n62% prefer hybrid\n24% prefer fully remote\n14% prefer on-site\n"
              "Source: 4,200 respondents",
         q="What percentage of respondents prefer fully remote work on survey_infographic.png?", a="24"),
    dict(cat="infographic", id="sales_infographic.png",
         text="Q1 REGIONAL SALES\nNorth: $1.2M\nSouth: $0.9M\nEast: $1.5M\nWest: $1.1M",
         q="Which region had the highest sales on sales_infographic.png?", a="East"),
    # visualmrc — dense passage + reading comprehension: answer is a phrase located in the passage
    dict(cat="visualmrc", id="article_climate.png",
         text="Coral reefs support roughly a quarter of all marine species despite covering less than 1% "
              "of the ocean floor. The primary driver of recent bleaching events is rising sea surface "
              "temperature, which disrupts the symbiosis between coral and algae.",
         q="According to article_climate.png, what is the primary driver of recent bleaching events?",
         a="rising sea surface temperature"),
    dict(cat="visualmrc", id="article_history.png",
         text="The Bridgewater Canal, opened in 1761, was financed largely by the Duke of Bridgewater to "
              "transport coal from his mines at Worsley to Manchester. It is often cited as the first true "
              "canal of the British Industrial Revolution.",
         q="According to article_history.png, who financed the Bridgewater Canal?", a="Duke of Bridgewater"),
    # chart — textual chart: read a series value / compare to find the peak month
    dict(cat="chart", id="bar_chart.png",
         text="Monthly Active Users (thousands)\nJanuary: 120\nFebruary: 135\nMarch: 158\nApril: 142",
         q="How many thousand monthly active users were there in March on bar_chart.png?", a="158"),
    dict(cat="chart", id="line_chart.png",
         text="Website Latency by Month (ms)\nMay: 220\nJune: 190\nJuly: 260\nAugust: 175",
         q="Which month had the highest latency on line_chart.png?", a="July"),
]

# OCR-as-tool + answer modeled as tools, so the scaffold can render/parse the call in any syntax.
_TOOLS = [Tool("ocr", "image_id", "returns the text an OCR encoder extracts from that image"),
          Tool("answer", "value", "submit your final answer — just the value")]

_NORM_RE = re.compile(r"[\s$,]")


def _norm(s: str) -> str:
    # strip surrounding sentence punctuation too (real VQA golds are terse like "Soil." / "Land.") —
    # but only at the ends, so internal dots survive for float golds ("42.50").
    return _NORM_RE.sub("", s.strip().lower()).strip(".!?")


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
        for i, d in enumerate(docs):
            fmt = resolve_format(i, scaffold)
            instr = ("Use the available tools to read the image, then answer." if fmt == "native"
                     else render_instructions(_TOOLS, fmt))
            # State the image id explicitly: real corpus questions don't name the file, so without this
            # the model has no handle to pass to ocr() and the loop is impossible (it just asks for the id).
            question = (f"You have a task about an IMAGE (id: {d['id']}) you cannot see directly.\n"
                        f"Call ocr on that id to read its text, then answer.\n\nTask: {d['q']}\n\n{instr}")
            rows.append({"question": question, "answer": d["a"],
                         "info": {"image_id": d["id"], "text": d["text"], "answer": d["a"],
                                  "category": d["cat"], "fmt": fmt}})
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
                     scaffold: str = "mixed", docs_path: str = "", **kwargs) -> vf.Environment:
    """scaffold: 'line'|'xml'|'json'|'poolside' (text syntax) · 'mixed' (round-robin) · 'native' (structured tool_calls).
    docs_path: a {cat,id,text,q,a} JSONL pack (real corpus questions + noisy OCR transcripts as `text`,
    built by `ocr_backend_eval build-docs`) — the end-to-end real-OCR loop. Empty = the mock perfect-text builtin docs."""
    if docs_path:
        docs = [json.loads(line) for line in Path(docs_path).read_text().splitlines() if line.strip()]
    else:
        docs = _BUILTIN_DOCS
    return OCRToolEnv(docs, max_turns=max_turns, efficiency_weight=efficiency_weight, scaffold=scaffold)
