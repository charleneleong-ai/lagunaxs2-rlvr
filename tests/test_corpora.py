import pytest

from laguna_rlvr.visual.corpora import build_corpus
from laguna_rlvr.visual.data import SyntheticOCR


def test_build_corpus_dispatches_synthetic():  # offline — no network/model
    ds = build_corpus("synthetic", 8)
    assert isinstance(ds, SyntheticOCR) and len(ds) == 8


def test_build_corpus_unknown_raises():
    with pytest.raises(ValueError):
        build_corpus("nope", 4)


def test_mixture_blends_corpora_by_weight():  # offline — synthetic only
    from laguna_rlvr.visual.corpora import _Mixture

    mix = _Mixture([("synthetic", 0.75), ("synthetic", 0.25)], n=8)
    assert len(mix) == 8  # round(8*.75)=6 + round(8*.25)=2
    img, txt = mix[0]
    assert txt  # yields (image, text) like any corpus
