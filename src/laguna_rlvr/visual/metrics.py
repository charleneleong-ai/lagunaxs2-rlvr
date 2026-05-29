from __future__ import annotations

import jiwer


def cer(pred: str, ref: str) -> float:
    """Character error rate. 0.0 = exact; ~1.0 = unrelated. Empty ref -> 0 if pred empty else 1."""
    if not ref:
        return 0.0 if not pred else 1.0
    return jiwer.cer(reference=ref, hypothesis=pred)
