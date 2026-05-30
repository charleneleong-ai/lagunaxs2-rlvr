"""Design2Code as (screenshot, HTML) pairs — the fixed held-out eval for ranking mixtures.

484 real C4 webpages (SALT-NLP/Design2Code, arXiv 2403.03163), disjoint from every training corpus —
so every mixture variant is scored on the *same unseen* set (the apples-to-apples ranker, AutoMixer
§3.2.3). The HF parquet exposes only the screenshot, so we fetch the repo snapshot and pair each
{id}.png with its {id}.html. Eval only — never put this in the training mix (leakage).
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import Dataset


class Design2Code(Dataset):
    """(screenshot, source-HTML) pairs from the Design2Code repo snapshot."""

    def __init__(self, n: int | None = 128, max_html_chars: int = 8192):
        root = Path(snapshot_download("SALT-NLP/Design2Code", repo_type="dataset"))
        pairs: list[tuple[Path, str]] = []
        for html in sorted(root.glob("*.html")):
            png = html.with_suffix(".png")
            if png.exists():
                pairs.append((png, html.read_text(errors="ignore")[:max_html_chars]))
                if n is not None and len(pairs) >= n:
                    break
        if not pairs:
            raise RuntimeError("no Design2Code {id}.png/{id}.html pairs found in the snapshot")
        self.items = pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Image.Image, str]:
        png, html = self.items[i]
        return Image.open(png).convert("RGB"), html
