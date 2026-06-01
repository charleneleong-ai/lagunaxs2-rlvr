"""ScreenSpot-v2 — GUI grounding over 1272 cleaned mobile/desktop/web screenshots (lmms-lab/ScreenSpot-v2,
arXiv 2401.10935 + the v2 relabel). The bbox ships as `[x, y, w, h]` in PIXELS, so we convert to
`[x1, y1, x2, y2]` normalized to 0-1 by the image width/height and format the locate prompt + target
exactly like hf_image_text.ScreenSpotDataset, so the triples plug straight into grounding.screenspot_eval
(IoU@0.5 / center / parse-rate). Reuses that scorer — no duplicate IoU logic here.
"""
from __future__ import annotations

from itertools import islice

from datasets import Dataset as HFDataset
from datasets import Features
from datasets import Image as HFImage
from datasets import Value, load_dataset
from PIL import Image
from rich.progress import track
from torch.utils.data import Dataset

from laguna_rlvr.visual.hf_image_text import _cached_or_stream


def _to_norm_box(bbox_xywh, img_w: int, img_h: int) -> str:
    """Pixel `[x, y, w, h]` -> `"[x1, y1, x2, y2]"` normalized to 0-1 by the image dimensions."""
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h
    return "[" + ", ".join(f"{c:.3f}" for c in (x1, y1, x2, y2)) + "]"


class ScreenSpotV2Dataset(Dataset):
    """(image, locate-question, normalized-box-text) from lmms-lab/ScreenSpot-v2 — streamed + cached.
    Box-text is `[x1, y1, x2, y2]` in 0-1, so it scores with grounding.screenspot_eval directly."""

    def __init__(self, repo: str = "lmms-lab/ScreenSpot-v2", *, split: str = "train", n: int = 1000,
                 offset: int = 0):
        key = "screenspot_v2__" + "__".join(str(p) for p in (repo, split, n, offset))
        self._ds = _cached_or_stream(key, lambda: self._stream(repo, split, n, offset))

    @staticmethod
    def _stream(repo, split, n, offset) -> HFDataset:
        stream = load_dataset(repo, split=split, streaming=True)
        imgs, qs, ans = [], [], []
        for row in track(islice(stream, offset, offset + n), total=n, description=f"{repo} ({n})"):
            img, instr, box = row.get("image"), row.get("instruction"), row.get("bbox")
            if img is not None and instr and box and len(box) == 4:
                img = img.convert("RGB")
                imgs.append(img)
                qs.append(f"Locate the UI element for '{instr}'. Give its bounding box as [x1, y1, x2, y2] in 0-1.")
                ans.append(_to_norm_box(box, img.width, img.height))
        if not imgs:
            raise RuntimeError(f"no usable rows from {repo}")
        return HFDataset.from_dict(
            {"image": imgs, "question": qs, "answer": ans},
            features=Features({"image": HFImage(), "question": Value("string"), "answer": Value("string")}))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, str, str]:
        row = self._ds[i]
        return row["image"], row["question"], row["answer"]
