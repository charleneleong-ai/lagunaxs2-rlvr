import pytest

from laguna_rlvr.visual.corpora import build_corpus
from laguna_rlvr.visual.data import SyntheticOCR


def test_build_corpus_dispatches_synthetic():  # offline — no network/model
    ds = build_corpus("synthetic", 8)
    assert isinstance(ds, SyntheticOCR) and len(ds) == 8


def test_build_corpus_unknown_raises():
    with pytest.raises(ValueError):
        build_corpus("nope", 4)


def test_parse_mixture():
    from laguna_rlvr.visual.corpora import parse_mixture

    assert parse_mixture("websight=0.6, webcode2m=0.4") == [("websight", 0.6), ("webcode2m", 0.4)]


def test_mixture_blends_corpora_by_weight():  # offline — synthetic only
    from laguna_rlvr.visual.corpora import _Mixture

    mix = _Mixture([("synthetic", 0.75), ("synthetic", 0.25)], n=8)
    assert len(mix) == 8  # round(8*.75)=6 + round(8*.25)=2
    img, txt, corpus = mix[0]
    assert txt and corpus == "synthetic"  # yields (image, text, corpus-tag)


def test_hf_image_text_caches_to_disk(tmp_path, monkeypatch):  # offline — load_dataset mocked
    from PIL import Image

    import laguna_rlvr.visual.hf_image_text as hit

    monkeypatch.setattr(hit, "_CACHE_DIR", tmp_path / "cache")
    rows = [{"image": Image.new("RGB", (8, 8), c), "text": f"<p>{c}</p>"} for c in ("red", "blue")]
    monkeypatch.setattr(hit, "load_dataset", lambda *a, **k: iter(rows))
    first = hit.HFImageTextDataset("fake/repo", n=2)
    assert len(first) == 2 and first[0][1] == "<p>red</p>"

    # second build must hit the disk cache — load_dataset now raises if the network is touched
    def _boom(*a, **k):
        raise AssertionError("re-streamed despite cache")

    monkeypatch.setattr(hit, "load_dataset", _boom)
    second = hit.HFImageTextDataset("fake/repo", n=2)
    assert [second[i][1] for i in range(len(second))] == ["<p>red</p>", "<p>blue</p>"]
