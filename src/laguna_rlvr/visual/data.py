from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

_DEFAULT_PHRASES = [
    "Invoice 2026", "Total: $1,240.50", "The quick brown fox",
    "Section 4.2 Results", "shipped 2026-05-29", "Account #00831",
]


def render_text(text: str, size: tuple[int, int] = (384, 96), seed: int | None = None) -> Image.Image:
    """Render `text` onto a white RGB image. The label of this image is `text` itself."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()                      # always available; no font files needed
    draw.text((rng.randint(4, 20), rng.randint(4, 20)), text, fill="black", font=font)
    return img


class SyntheticOCR(Dataset):
    """(image, exact-text) pairs. The label is ground truth by construction — self-verifying."""

    def __init__(self, texts: list[str] | None = None, n: int | None = None, seed: int = 0):
        rng = random.Random(seed)
        if texts is None:
            texts = [rng.choice(_DEFAULT_PHRASES) + f" {rng.randint(0, 9999)}" for _ in range(n or 256)]
        self.texts = texts
        self.seed = seed

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> tuple[Image.Image, str]:
        return render_text(self.texts[i], seed=self.seed + i), self.texts[i]
