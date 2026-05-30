"""Stage-0 baselines: what does text-only Laguna do on the visual tasks *before* any adapter?

Two no-adapter baselines over the frozen base LLM — `blind` (task prompt only) and tool-mediated
(`GLM-OCR → text → Laguna`) — give the adapter's eventual numbers a floor and a bar to beat. See
docs/specs/2026-05-30-stage-0-baseline-design.md.
"""
from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModelForImageTextToText

from laguna_rlvr.visual.encoders import _REPOS  # canonical model-id registry (avoid a 2nd source of truth)


@torch.no_grad()
def glm_ocr_transcribe(items: list, device: str = "cuda", max_new_tokens: int = 48,
                       model=None, proc=None) -> list[str]:
    """GLM-OCR reads each item's image (image -> OCR text). One transcript per item, in order.

    Loads GLM-OCR once when `model`/`proc` are omitted, so calling this once with the full item list
    is the cheap path (the staged-GPU harness frees it before loading Laguna). `items` are
    (image, ...) tuples; only `it[0]` is read.
    """
    if model is None:
        repo = _REPOS["glm_ocr"]
        model = AutoModelForImageTextToText.from_pretrained(repo, device_map=device).eval()
        proc = AutoImageProcessor.from_pretrained(repo)
    out = []
    for it in items:
        batch = proc(images=[it[0]], return_tensors="pt").to(model.device)
        gen = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        out.append(proc.batch_decode(gen, skip_special_tokens=True)[0])
    return out
