import numpy as np
from PIL import Image
from laguna_rlvr.visual.corpora import load_text_image
from laguna_rlvr.visual.data import _ALPHANUM, random_words, render_text, SyntheticOCR


def _ink(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L")) < 128  # boolean mask of dark (text) pixels


def test_render_returns_rgb_image():
    img = render_text("hello world")
    assert isinstance(img, Image.Image) and img.mode == "RGB"


def test_random_words_deterministic_fixed_length_varied():
    a = random_words(64, seed=0, length=4)
    assert a == random_words(64, seed=0, length=4)            # seeded -> reproducible
    assert all(len(w) == 4 and set(w) <= set(_ALPHANUM) for w in a)
    assert len(set(a)) > 1                                    # not a single repeated string


def test_large_centered_render_is_bigger_and_centered():
    big, small = _ink(render_text("AB", (384, 384), font_size=72, center=True)), _ink(render_text("AB"))
    assert big.sum() > small.sum() * 4                        # font_size -> many more ink pixels
    cols = np.where(big.any(axis=0))[0]
    assert abs((384 - cols[-1]) - cols[0]) < 40               # left/right margins ~equal -> centered


def test_dataset_label_is_exact_source_text():
    ds = SyntheticOCR(texts=["abc", "def 123"], seed=0)
    img, label = ds[1]
    assert label == "def 123"               # label is the exact source string
    assert isinstance(img, Image.Image)


def test_dataset_len():
    assert len(SyntheticOCR(texts=["a", "b", "c"], seed=0)) == 3


def test_synthetic_words_lg_yields_big_glyph_reading_pairs():
    ds = load_text_image("synthetic_words_lg", 8)
    img, label = ds[0]
    assert len(label) == 4 and _ink(img).mean() > 0.01     # 4-char target, glyphs actually rendered
