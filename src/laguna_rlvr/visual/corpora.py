"""Registry mapping a `--dataset` name to a builder `build(n) -> Dataset`.

Training corpora for the projector. Builders import lazily so a corpus's heavy deps (HF `datasets`,
image downloads) only load when that corpus is requested. The train/val split is applied by the
caller (`train.py`, seeded 90/10). Held-out *eval* sets (Design2Code, SWE-bench M test via the
agentic verifier) are intentionally not training corpora — see docs/a100-multimodal-adapter.md.
"""
from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import Dataset


def _synthetic(n: int) -> Dataset:
    from laguna_rlvr.visual.data import SyntheticOCR

    return SyntheticOCR(n=n)


def _swebench_mm(n: int) -> Dataset:
    from laguna_rlvr.visual.swebench_mm import SWEBenchMultimodal

    return SWEBenchMultimodal()  # full 612-instance set (dev+test); ignores n


def _websight(n: int) -> Dataset:
    from laguna_rlvr.visual.hf_image_text import HFImageTextDataset

    return HFImageTextDataset("HuggingFaceM4/WebSight", config="v0.1", n=n)  # screenshot -> HTML/CSS


def _webcode2m(n: int) -> Dataset:
    from laguna_rlvr.visual.hf_image_text import HFImageTextDataset

    return HFImageTextDataset("xcodemind/webcode2m", n=n)  # real webpage design -> code


REGISTRY: dict[str, Callable[[int], Dataset]] = {
    "synthetic": _synthetic,
    "swebench_mm": _swebench_mm,
    "websight": _websight,
    "webcode2m": _webcode2m,
}


def build_corpus(name: str, n: int) -> Dataset:
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choices: {', '.join(REGISTRY)}")
    return REGISTRY[name](n)
