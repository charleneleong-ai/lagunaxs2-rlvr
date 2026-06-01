"""OCRBench — the 1000-item OCR benchmark (echo840/OCRBench, arXiv 2305.07895), five task families
(text recognition, scene/document VQA, KIE, handwritten math). Eval only — disjoint from training.

Streamed + disk-cached like the other HF loaders. The scorer follows the official OCRBench protocol:
a prediction is correct if ANY gold answer (lowercased, stripped) is a substring of the lowercased
prediction — EXCEPT the "Handwritten Mathematical Expression Recognition" category, which is
whitespace-removed exact match (the LaTeX answer must match verbatim modulo spaces). Returns overall
accuracy plus a per-category breakdown.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import islice

from datasets import Dataset as HFDataset
from datasets import Features
from datasets import Image as HFImage
from datasets import Sequence, Value, load_dataset
from PIL import Image
from rich.progress import track
from torch.utils.data import Dataset

from laguna_rlvr.visual.hf_image_text import _cached_or_stream
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter

_HME = "Handwritten Mathematical Expression Recognition"


class OCRBenchDataset(Dataset):
    """(image, question, answers, category) from echo840/OCRBench — streamed + disk-cached."""

    def __init__(self, repo: str = "echo840/OCRBench", *, split: str = "test", n: int = 1000,
                 offset: int = 0):
        key = "ocrbench__" + "__".join(str(p) for p in (repo, split, n, offset))
        self._ds = _cached_or_stream(key, lambda: self._stream(repo, split, n, offset))

    @staticmethod
    def _stream(repo, split, n, offset) -> HFDataset:
        stream = load_dataset(repo, split=split, streaming=True)
        imgs, qs, ans, cats = [], [], [], []
        for row in track(islice(stream, offset, offset + n), total=n, description=f"{repo} ({n})"):
            img, q, a = row.get("image"), row.get("question"), row.get("answer")
            cat = row.get("dataset") or row.get("question_type")
            if img is not None and q and a:
                imgs.append(img.convert("RGB"))
                qs.append(q)
                ans.append([str(x) for x in (a if isinstance(a, list) else [a])])
                cats.append(cat or "unknown")
        if not imgs:
            raise RuntimeError(f"no usable rows from {repo}")
        return HFDataset.from_dict(
            {"image": imgs, "question": qs, "answer": ans, "category": cats},
            features=Features({"image": HFImage(), "question": Value("string"),
                               "answer": Sequence(Value("string")), "category": Value("string")}))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int) -> tuple[Image.Image, str, list[str], str]:
        row = self._ds[i]
        return row["image"], row["question"], row["answer"], row["category"]


def _ocrbench_correct(prediction: str, answers: list[str], category: str) -> bool:
    """OCRBench match: any gold answer (lowercased, stripped) is a substring of the lowercased
    prediction; HME is whitespace-removed exact match against any gold answer instead."""
    pred = prediction.lower()
    if category == _HME:
        p = "".join(pred.split())
        return any(p == "".join(a.lower().split()) for a in answers)
    return any(a.lower().strip() in pred for a in answers)


def _slug(category: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in category.lower()).strip("_")


def ocrbench_eval(adapter: VisualAdapter, items, max_new_tokens: int = 48,
                  prefix: str = "ocrbench") -> dict[str, float]:
    """`items`: (image, question, answers, category). Ask the adapter each question, score by the
    OCRBench protocol, return overall accuracy + per-category accuracy."""
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for img, question, answers, category in items:
        reply = adapter.chat([Turn(f"{IMAGE_TOKEN}\n{question}", [img])], max_new_tokens=max_new_tokens)[0]
        per_cat[category][0] += int(_ocrbench_correct(reply, answers, category))
        per_cat[category][1] += 1
    hits, total = (sum(c) for c in zip(*per_cat.values())) if per_cat else (0, 0)
    out = {f"{prefix}/metrics/accuracy": hits / max(total, 1)}
    out.update({f"{prefix}/metrics/acc_{_slug(c)}": h / max(t, 1) for c, (h, t) in per_cat.items()})
    return out
