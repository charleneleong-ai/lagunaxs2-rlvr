import pytest

from laguna_rlvr.visual.corpora import build_corpus
from laguna_rlvr.visual.data import SyntheticOCR


def test_build_corpus_dispatches_synthetic():  # offline — no network/model
    ds = build_corpus("synthetic", 8)
    assert isinstance(ds, SyntheticOCR) and len(ds) == 8


def test_build_corpus_unknown_raises():
    with pytest.raises(ValueError):
        build_corpus("nope", 4)
