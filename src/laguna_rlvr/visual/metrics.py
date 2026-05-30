from __future__ import annotations

import jiwer


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


def transcription_metrics(adapter, items: list, prefix: str = "val") -> dict[str, float]:
    """Transcribe each image and score mean WER/CER vs its label — generation quality, not loss.

    `items` are (image, label, ...) tuples; `adapter` exposes `transcribe(list[image]) -> list[str]`.
    Generation-based (slow) — call on a small subset. `prefix` namespaces the keys (e.g. val / eval).
    """
    if not items:
        return {}
    # one image per call: the encoder can't stack variable-resolution images into a batch.
    preds = [adapter.transcribe([it[0]])[0] for it in items]
    refs = [it[1] for it in items]
    n = len(items)
    return {f"{prefix}/wer": sum(wer(p, r) for p, r in zip(preds, refs)) / n,
            f"{prefix}/cer": sum(cer(p, r) for p, r in zip(preds, refs)) / n}
