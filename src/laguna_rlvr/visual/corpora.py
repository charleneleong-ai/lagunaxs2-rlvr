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


def _chartmimic(n: int) -> Dataset:
    from laguna_rlvr.visual.hf_image_text import HFImageTextDataset

    # chart image -> matplotlib code (re-renderable, verifiable). ChartMimic is an eval benchmark,
    # so exclude these instances from any chart eval to avoid leakage (cf. swebench_mm).
    return HFImageTextDataset(
        "ChartMimic/ChartMimic", config="chartmimic", split="test", n=n,
        image_col="GroundTruthFigurePreview", text_col="GroundTruthFigureCode")


class _Mixture(Dataset):
    """Weighted blend of corpora for full training — builds ~n×weight examples from each and
    concatenates them into one indexable dataset, so the model sees the corpora interleaved (the
    seeded 90/10 train/val split + DataLoader shuffle then mix them). This is the projector-stage
    analog of the report's pre-training data mixture (§3.2.3 AutoMixer / Table 4): the model learns
    the full corpus mix in one run rather than one dataset at a time.

    Note: swebench_mm is fixed-size (612) and ignores its quota — negligible when the mix is large.
    """

    def __init__(self, specs: list[tuple[str, float]], n: int):
        total = sum(w for _, w in specs)
        self._datasets: list[Dataset] = []
        self._index: list[tuple[int, int]] = []
        for di, (name, weight) in enumerate(specs):
            ds = build_corpus(name, max(1, round(n * weight / total)))
            self._datasets.append(ds)
            self._index += [(di, j) for j in range(len(ds))]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        di, j = self._index[i]
        return self._datasets[di][j]


# Default full-training mixture (WebSight-heavy; mirrors the corpus plan in the design doc).
_DEFAULT_MIX = [("websight", 0.55), ("webcode2m", 0.25), ("chartmimic", 0.1), ("swebench_mm", 0.1)]


def _mix(n: int) -> Dataset:
    return _Mixture(_DEFAULT_MIX, n)


REGISTRY: dict[str, Callable[[int], Dataset]] = {
    "synthetic": _synthetic,
    "swebench_mm": _swebench_mm,
    "websight": _websight,
    "webcode2m": _webcode2m,
    "chartmimic": _chartmimic,
    "mix": _mix,
}


def build_corpus(name: str, n: int) -> Dataset:
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choices: {', '.join(REGISTRY)}")
    return REGISTRY[name](n)
