from __future__ import annotations

import jiwer

from laguna_rlvr.visual.code_metrics import code_validity_rate, codebleu_score
from laguna_rlvr.visual.corpora import CORPUS_KIND


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
    # one image per call: the encoder can't stack variable-resolution images into a batch.
    preds = [adapter.transcribe([it[0]])[0] for it in items]
    refs = [it[1] for it in items]
    n = len(items)
    out = {f"{prefix}/metrics/wer": sum(wer(p, r) for p, r in zip(preds, refs)) / n,
           f"{prefix}/metrics/cer": sum(cer(p, r) for p, r in zip(preds, refs)) / n}
    # code-validity (no-exec) + CodeBLEU (structural), keyed off the corpus tag (3rd element, present
    # for the mix) -> code kind; skipped when no item has a code target.
    kinds = [CORPUS_KIND.get(it[2]) if len(it) > 2 else None for it in items]
    if (rate := code_validity_rate(preds, kinds)) is not None:
        out[f"{prefix}/metrics/code_valid"] = rate
    if (cb := codebleu_score(preds, refs, kinds)) is not None:
        out[f"{prefix}/metrics/codebleu"] = cb
    return out
