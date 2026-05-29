from PIL import Image
from laguna_rlvr.visual.data import render_text, SyntheticOCR


def test_render_returns_rgb_image():
    img = render_text("hello world")
    assert isinstance(img, Image.Image) and img.mode == "RGB"


def test_dataset_label_is_exact_source_text():
    ds = SyntheticOCR(texts=["abc", "def 123"], seed=0)
    img, label = ds[1]
    assert label == "def 123"               # label is the exact source string
    assert isinstance(img, Image.Image)


def test_dataset_len():
    assert len(SyntheticOCR(texts=["a", "b", "c"], seed=0)) == 3
