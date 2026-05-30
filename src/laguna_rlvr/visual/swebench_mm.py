"""SWE-bench Multimodal as (screenshot, issue-text) pairs for projector alignment on real
visual-software artifacts.

The dataset's `image_assets.problem_statement` field holds the screenshots embedded in each GitHub
issue (real chart/UI bug renders from JS libraries); we pair each with the cleaned issue text. This
is stage-1.5 real-data projector SFT (vs the synthetic OCR scaffold). The downstream objective is
the agentic see->fix->verify loop scored by the FAIL_TO_PASS verifier (see docs); this module only
supplies the alignment corpus + the images.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx
from datasets import load_dataset
from PIL import Image
from rich.progress import track
from torch.utils.data import Dataset

_CACHE = Path.home() / ".cache" / "swebench_mm_images"
_IMG_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # ![alt](url)
_HTML_IMG = re.compile(r"<img[^>]*>")
_WS = re.compile(r"\s+")


def _clean_caption(text: str, max_chars: int) -> str:
    """Strip image markup + collapse whitespace so the label is the issue's prose, not URLs."""
    text = _HTML_IMG.sub("", _IMG_MD.sub("", text))
    return _WS.sub(" ", text).strip()[:max_chars]


def _download(url: str, timeout: float = 12.0) -> Path | None:
    """Fetch + validate an image, cached by URL hash. Returns None on any failure (dead link etc.)."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    dst = _CACHE / f"{hashlib.sha1(url.encode()).hexdigest()}.img"
    if dst.exists():
        return dst  # already fetched + validated on first download
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        dst.write_bytes(r.content)
        Image.open(dst).convert("RGB")  # validate decodable before trusting the cache
        return dst
    except (httpx.HTTPError, OSError):
        dst.unlink(missing_ok=True)
        return None


class SWEBenchMultimodal(Dataset):
    """(screenshot, cleaned-issue-text) pairs from SWE-bench Multimodal problem statements."""

    # NOTE: "dev+test" is the full 612-instance set. `test` (510) is SWE-bench M's held-out
    # leaderboard split — using it as projector-alignment data leaks if the agentic verifier is
    # later evaluated on it. Acceptable for image->issue-text alignment; exclude before RL eval.
    def __init__(self, split: str = "dev+test", n: int | None = None, max_caption_chars: int = 240):
        rows = load_dataset("SWE-bench/SWE-bench_Multimodal", split=split)
        items: list[tuple[Path, str]] = []
        for row in track(rows, description="SWE-bench MM images"):
            try:
                urls = json.loads(row["image_assets"]).get("problem_statement", [])
            except (json.JSONDecodeError, TypeError):
                continue
            caption = _clean_caption(row["problem_statement"], max_caption_chars)
            for url in urls:
                if (path := _download(url)) is not None:
                    items.append((path, caption))
            if n is not None and len(items) >= n:
                break
        if not items:
            raise RuntimeError("no SWE-bench MM images could be downloaded (network?)")
        self.items = items[:n] if n is not None else items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Image.Image, str]:
        path, caption = self.items[i]
        return Image.open(path).convert("RGB"), caption
