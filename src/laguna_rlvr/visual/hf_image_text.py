"""Stream an HF (image, text) dataset into (screenshot, code) pairs.

For corpora whose screenshot is an embedded `Image` column — WebSight, WebCode2M — so there is no
URL download (unlike SWE-bench M). These sets are huge (WebSight 31GB+, WebCode2M ~1TB), so we
`streaming=True` and materialize only the first `n` rows rather than downloading the whole parquet.

The label is the page's HTML/code, truncated to keep projector-SFT sequences bounded; the full
screenshot->code objective with untruncated code belongs to the later long-context / RLVR stage.
"""
from __future__ import annotations

from itertools import islice

from datasets import load_dataset
from rich.progress import track
from torch.utils.data import Dataset


class HFImageTextDataset(Dataset):
    """(screenshot, code) pairs streamed from an HF dataset with embedded image + text columns."""

    def __init__(self, repo: str, *, config: str | None = None, split: str = "train",
                 n: int = 2000, offset: int = 0, image_col: str = "image", text_col: str = "text",
                 max_text_chars: int = 2048):
        stream = load_dataset(repo, config, split=split, streaming=True)
        items: list[tuple] = []
        # offset skips the first `offset` rows — used to carve a held-out eval slice disjoint from
        # the training range (which streams from row 0).
        for row in track(islice(stream, offset, offset + n), total=n, description=f"{repo} ({n})"):
            img, txt = row.get(image_col), row.get(text_col)
            if img is not None and txt:
                items.append((img.convert("RGB"), txt[:max_text_chars]))
        if not items:
            raise RuntimeError(f"no usable rows from {repo} (cols {image_col!r}/{text_col!r})")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]
