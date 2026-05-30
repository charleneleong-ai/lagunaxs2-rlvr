"""Simulated, verifiable multi-turn multimodal QA eval.

No off-the-shelf multi-turn *multimodal* QA set exists for our (code/UI) domain with ground truth, so
we synthesize one from SyntheticOCR — we *know* each image's text. Each episode shows two images,
asks to read each, then a text-only follow-up that requires recalling the FIRST image. We score
per-turn reading (`qa/accuracy`) AND cross-turn memory (`qa/recall`) by normalized-substring match.

This measures whether a *single-turn-trained* projector transfers to multi-turn (agentic) use via the
frozen LLM + `<image>` splice. A gap motivates Stage-2 multi-turn training (agentic SFT, report §4.3.3).
"""
from __future__ import annotations

import re

from laguna_rlvr.visual.data import render_text
from laguna_rlvr.visual.model import _PROMPT, Turn, VisualAdapter


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def evaluate_multiturn_qa(adapter: VisualAdapter, n: int = 16, seed: int = 0,
                          max_new_tokens: int = 24) -> dict[str, float]:
    """Run n 3-turn episodes (read img A, read img B, recall img A) and score by substring match."""
    hits = total = recall_hits = 0
    for i in range(n):
        a, b = f"invoice {seed + i}", f"total {seed + 7 * i + 3}"  # distinct ground truth per episode
        turns = [
            Turn(_PROMPT, [render_text(a, seed=seed + i)]),
            Turn(_PROMPT, [render_text(b, seed=seed + i + 1000)]),  # +1000: distinct render from A
            Turn("What text was in the first image?"),  # cross-turn recall (text-only turn)
        ]
        r1, r2, r3 = adapter.chat(turns, max_new_tokens=max_new_tokens)
        # substring (not CER): the reply is verbose ("the text is …"); we score whether it CONTAINS
        # the ground truth, which CER would wrongly penalize as edits.
        for expected, reply in ((a, r1), (b, r2)):
            total += 1
            hits += _norm(expected) in _norm(reply)
        recall_hits += _norm(a) in _norm(r3)
    return {"qa/accuracy": hits / max(total, 1), "qa/recall": recall_hits / max(n, 1)}
