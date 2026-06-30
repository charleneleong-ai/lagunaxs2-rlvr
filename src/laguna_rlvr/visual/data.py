from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

_DEFAULT_PHRASES = [
    "Invoice 2026", "Total: $1,240.50", "The quick brown fox",
    "Section 4.2 Results", "shipped 2026-05-29", "Account #00831",
]


def render_text(text: str, size: tuple[int, int] = (384, 96), seed: int | None = None,
                font_size: int | None = None, center: bool = False) -> Image.Image:
    """Render `text` onto a white RGB image. The label of this image is `text` itself.

    `font_size` selects a scalable font (big glyphs span many patches = high SNR for the encoder-free
    reading diagnostic); `center` places the text mid-canvas so resize+center-crop won't clip it."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=font_size) if font_size else ImageFont.load_default()
    if center:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        xy = ((size[0] - (right - left)) // 2 - left, (size[1] - (bottom - top)) // 2 - top)
    else:
        xy = (rng.randint(4, 20), rng.randint(4, 20))
    draw.text(xy, text, fill="black", font=font)
    return img


_ALPHANUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def random_words(n: int, seed: int, length: int = 4) -> list[str]:
    """`n` random fixed-length alphanumeric strings — infinite vocab, so reading them is genuine
    transcription (no closed-set memorization / LM-prior shortcut)."""
    rng = random.Random(seed)
    return ["".join(rng.choice(_ALPHANUM) for _ in range(length)) for _ in range(n)]


class SyntheticOCR(Dataset):
    """(image, exact-text) pairs. The label is ground truth by construction — self-verifying."""

    def __init__(self, texts: list[str] | None = None, n: int | None = None, seed: int = 0,
                 size: tuple[int, int] = (384, 96), font_size: int | None = None, center: bool = False):
        rng = random.Random(seed)
        if texts is None:
            texts = [rng.choice(_DEFAULT_PHRASES) + f" {rng.randint(0, 9999)}" for _ in range(n or 256)]
        self.texts = texts
        self.seed = seed
        self._render = dict(size=size, font_size=font_size, center=center)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> tuple[Image.Image, str]:
        return render_text(self.texts[i], seed=self.seed + i, **self._render), self.texts[i]
