"""GUI-grounding eval — parse a predicted bounding box from the model's text and score it against the
ground-truth box by IoU@0.5 and center-in-box (the ScreenSpot metric). Turns ScreenSpotDataset into a
real *localize* eval; the reading metric (substring/F1) can't score a box. The 'act/locate' half of an
agentic vision model, alongside Read / Converse / Edit."""
from __future__ import annotations

import re

from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter

_BOX = re.compile(r"(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)")


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    """First [x1,y1,x2,y2] found in `text` (brackets optional). None if absent."""
    m = _BOX.search(text)
    return tuple(float(x) for x in m.groups()) if m else None


def box_iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_in_box(pred, true) -> bool:
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return true[0] <= cx <= true[2] and true[1] <= cy <= true[3]


def screenspot_eval(adapter: VisualAdapter, items: list, max_new_tokens: int = 32,
                    prefix: str = "ground") -> dict[str, float]:
    """`items`: (image, question, true-box-as-text). Ask the adapter to locate, parse its predicted
    box, score IoU@0.5 + center-in-box + parse-rate (did it even emit a box)."""
    iou_hits = center_hits = parsed = total = 0
    for img, question, true_text in items:
        true = parse_box(true_text)
        if true is None:
            continue
        total += 1
        reply = adapter.chat([Turn(f"{IMAGE_TOKEN}\n{question}", [img])], max_new_tokens=max_new_tokens)[0]
        if (pred := parse_box(reply)) is None:
            continue
        parsed += 1
        iou_hits += int(box_iou(pred, true) >= 0.5)
        center_hits += int(center_in_box(pred, true))
    n = max(total, 1)
    return {f"{prefix}/metrics/iou@0.5": iou_hits / n, f"{prefix}/metrics/center_acc": center_hits / n,
            f"{prefix}/metrics/parse_rate": parsed / n}
