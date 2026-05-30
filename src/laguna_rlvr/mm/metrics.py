from __future__ import annotations

import jiwer


def _error_rate(metric, pred: str, ref: str) -> float:
    """Shared empty-ref guard for jiwer cer/wer: empty ref -> 0 if pred empty else 1."""
    if not ref:
        return 0.0 if not pred else 1.0
    return metric(reference=ref, hypothesis=pred)


def cer(pred: str, ref: str) -> float:
    """Character error rate. 0.0 = exact; ~1.0 = unrelated."""
    return _error_rate(jiwer.cer, pred, ref)


def wer(pred: str, ref: str) -> float:
    """Word error rate (speech analog of CER)."""
    return _error_rate(jiwer.wer, pred, ref)
