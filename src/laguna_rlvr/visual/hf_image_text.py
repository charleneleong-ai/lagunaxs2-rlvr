"""Stream an HF (image, text) dataset into (screenshot, code) pairs, cached to disk after first fetch.

For corpora whose screenshot is an embedded `Image` column — WebSight, WebCode2M, ChartMimic — so there
is no URL download (unlike SWE-bench M). These sets are huge (WebSight 31GB+, WebCode2M ~1TB), so we
`streaming=True` and materialize only the first `n` rows rather than downloading the whole parquet.

The streamed rows are then saved to a compressed local cache (`datasets.save_to_disk`) keyed by the load
params, so every later run / restart loads from disk — no network, no mid-stream `ReadTimeout`. Point
`LAGUNA_DATA_CACHE` elsewhere to relocate it (default `~/.cache/laguna-mm/corpora`).

The label is the page's HTML/code, truncated to keep projector-SFT sequences bounded; the full
screenshot->code objective with untruncated code belongs to the later long-context / RLVR stage.
"""
from __future__ import annotations

import os
from itertools import islice
from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import Features
from datasets import Image as HFImage
from datasets import Value, load_dataset, load_from_disk
from rich.progress import track
from torch.utils.data import Dataset

_CACHE_DIR = Path(os.environ.get("LAGUNA_DATA_CACHE", Path.home() / ".cache" / "laguna-mm" / "corpora"))


class HFImageTextDataset(Dataset):
    """(screenshot, code) pairs streamed from an HF dataset (embedded image + text), cached to disk."""

    def __init__(self, repo: str, *, config: str | None = None, split: str = "train",
                 n: int = 2000, offset: int = 0, image_col: str = "image", text_col: str = "text",
                 max_text_chars: int = 2048):
        key = "__".join(str(p) for p in
                        (repo, config, split, n, offset, image_col, text_col, max_text_chars)).replace("/", "_")
        cache = _CACHE_DIR / key
        if cache.exists():
            self._ds = load_from_disk(str(cache))
        else:
            ds = self._stream(repo, config, split, n, offset, image_col, text_col, max_text_chars)
            cache.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache))
            self._ds = ds  # in-memory Arrow; kept lazy — decode per access, not all n images up front

    @staticmethod
    def _stream(repo: str, config: str | None, split: str, n: int, offset: int,
                image_col: str, text_col: str, max_text_chars: int) -> HFDataset:
        stream = load_dataset(repo, config, split=split, streaming=True)
        imgs, txts = [], []
        # offset skips the first `offset` rows — carves a held-out eval slice disjoint from the
        # training range (which streams from row 0).
        for row in track(islice(stream, offset, offset + n), total=n, description=f"{repo} ({n})"):
            img, txt = row.get(image_col), row.get(text_col)
            if img is not None and txt:
                imgs.append(img.convert("RGB"))
                txts.append(txt[:max_text_chars])
        if not imgs:
            raise RuntimeError(f"no usable rows from {repo} (cols {image_col!r}/{text_col!r})")
        # convert to RGB once here so the cached bytes are RGB — no per-access reconversion downstream.
        return HFDataset.from_dict({"image": imgs, "text": txts},
                                   features=Features({"image": HFImage(), "text": Value("string")}))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        row = self._ds[i]
        return row["image"], row["text"]
