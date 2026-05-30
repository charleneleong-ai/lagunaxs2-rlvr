from __future__ import annotations

import jiwer

from laguna_rlvr.visual.code_metrics import code_validity_rate, codebleu_score
from laguna_rlvr.visual.corpora import CORPUS_KIND

_OCR_GEN_TOKENS = 48    # short transcription targets
_CODE_GEN_TOKENS = 512  # code/HTML need room to form a compilable/parseable unit (matches label cap)


def _error_rate(pred: str, ref: str, fn) -> float:
    """A jiwer error rate with an empty-ref guard: 0.0 if pred is also empty, else 1.0."""
    if not ref:
        return 0.0 if not pred else 1.0
    return fn(reference=ref, hypothesis=pred)


def cer(pred: str, ref: str) -> float:
    """Character error rate. 0.0 = exact; ~1.0 = unrelated."""
    return _error_rate(pred, ref, jiwer.cer)


def wer(pred: str, ref: str) -> float:
    """Word error rate. 0.0 = exact; ~1.0 = unrelated."""
    return _error_rate(pred, ref, jiwer.wer)


def generation_metrics(adapter, items: list, prefix: str = "val") -> dict[str, float]:
    """Transcribe each image and score mean WER/CER (+ code-validity) vs its label — generation
    quality, not loss.

    `items` are (image, label, ...) tuples; `adapter` exposes `transcribe(list[image]) -> list[str]`.
    Generation-based (slow) — call on a small subset. `prefix` namespaces the keys (e.g. val / eval).
    """
    if not items:
        return {}
    kinds = [CORPUS_KIND.get(it[2]) if len(it) > 2 else None for it in items]
    # one image per call (variable-resolution images can't batch). Code targets get a longer budget —
    # a 48-token OCR cap truncates code mid-statement, making validity/codebleu meaningless.
    preds = [adapter.transcribe([it[0]], max_new_tokens=_OCR_GEN_TOKENS if k is None else _CODE_GEN_TOKENS)[0]
             for it, k in zip(items, kinds)]
    refs = [it[1] for it in items]
    out: dict[str, float] = {}
    # WER/CER are transcription metrics — meaningful only for OCR-style text targets (kind None).
    # On long code/HTML, exact-match explodes (WER >> 1), so code corpora ride on code_valid + codebleu.
    if ocr := [(p, r) for p, r, k in zip(preds, refs, kinds) if k is None]:
        out[f"{prefix}/metrics/wer"] = sum(wer(p, r) for p, r in ocr) / len(ocr)
        out[f"{prefix}/metrics/cer"] = sum(cer(p, r) for p, r in ocr) / len(ocr)
    # code-validity (no-exec) + CodeBLEU (structural) over code-kind targets; skipped when none.
    if (rate := code_validity_rate(preds, kinds)) is not None:
        out[f"{prefix}/metrics/code_valid"] = rate
    if (cb := codebleu_score(preds, refs, kinds)) is not None:
        out[f"{prefix}/metrics/codebleu"] = cb
    return out
