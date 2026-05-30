"""Registry mapping a `--dataset` name to a builder `build(n) -> Dataset`.

Training corpora for the projector. Builders import lazily so a corpus's heavy deps (HF `datasets`,
image downloads) only load when that corpus is requested. The train/val split is applied by the
caller (`train.py`, seeded 90/10). Held-out *eval* sets (Design2Code, SWE-bench M test via the
agentic verifier) are intentionally not training corpora — see docs/a100-multimodal-adapter.md.
"""
from __future__ import annotations

import re
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


def _design2code(n: int) -> Dataset:
    from laguna_rlvr.visual.design2code import Design2Code

    return Design2Code(n=n)  # EVAL ONLY (held-out external ranker) — never put in the training mix


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
        self._names: list[str] = []
        self._index: list[tuple[int, int]] = []
        for di, (name, weight) in enumerate(specs):
            quota = max(1, round(n * weight / total))
            ds = build_corpus(name, quota)
            self._datasets.append(ds)
            self._names.append(name)
            # cap to the quota so a fixed-size corpus (swebench_mm always returns 612) can't blow its
            # weight — the realized mix then matches the requested weights.
            self._index += [(di, j) for j in range(min(quota, len(ds)))]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        di, j = self._index[i]
        img, txt = self._datasets[di][j]
        return img, txt, self._names[di]  # 3rd element tags the corpus for per-corpus loss logging


# Default full-training mixture (WebSight-heavy; mirrors the corpus plan in the design doc). A
# hand-set prior — sweep it (scripts/mixture_sweep.py) and pick by held-out val rather than trust it.
# `synthetic` (SyntheticOCR) anchors the GLM-OCR encoder's native text-transcription path: without an
# OCR target the projector — the only trainable bridge — is free to repurpose those dims for the code
# objective and erode readout (multi-turn QA floored at 0 on the code-only mix, 2026-05). kind=None,
# so it's the corpus WER/CER scores against (vs the meaningless code WER it scored before).
_DEFAULT_MIX = [("websight", 0.45), ("webcode2m", 0.25), ("chartmimic", 0.1),
                ("swebench_mm", 0.1), ("synthetic", 0.1)]

REGISTRY: dict[str, Callable[[int], Dataset]] = {
    "synthetic": _synthetic,
    "swebench_mm": _swebench_mm,
    "websight": _websight,
    "webcode2m": _webcode2m,
    "chartmimic": _chartmimic,
    "design2code": _design2code,  # eval-only fixed held-out ranker (don't train on it)
}
CHOICES = [*REGISTRY, "mix"]

# Code "kind" of each corpus's targets, for code-validity metrics; corpora absent here aren't scored.
CORPUS_KIND = {"websight": "html", "webcode2m": "html", "design2code": "html", "chartmimic": "python"}

# Per-kind instruction a model is asked to follow (None = swebench-style issue text). Keyed by the
# values of CORPUS_KIND, so the corpus taxonomy and its task prompts live in one place.
TASK_PROMPT: dict[str | None, str] = {
    "html": "Write the HTML/CSS that renders this page.",
    "python": "Write the matplotlib code for this chart.",
    None: "Describe the software issue shown.",
}

_TITLE_PY = re.compile(r"""(?:set_title|suptitle|plt\.title)\(\s*['"]([^'"]+)['"]""")
_TITLE_HTML = re.compile(r"<title>(.*?)</title>", re.I | re.S)
_H1_HTML = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)


def extract_needle(label: str, kind: str | None) -> str | None:
    """The QA 'needle' for a corpus label: the chart/page title a model should read + later recall.
    Returns None when absent, so that row is skipped rather than scored against an empty answer.
    """
    if kind == "python":
        m = _TITLE_PY.search(label)
        return m.group(1).strip() if m else None
    if kind == "html":
        for pat in (_TITLE_HTML, _H1_HTML):
            if m := pat.search(label):
                if text := re.sub(r"<[^>]+>", "", m.group(1)).strip():  # drop tags nested in <h1>
                    return text
    return None


def parse_mixture(spec: str) -> list[tuple[str, float]]:
    """Parse a mixture string 'websight=0.6,webcode2m=0.4' into [(name, weight), ...]."""
    pairs = []
    for part in spec.split(","):
        name, _, weight = part.partition("=")
        pairs.append((name.strip(), float(weight)))
    return pairs


def build_corpus(name: str, n: int, mixture: list[tuple[str, float]] | None = None) -> Dataset:
    if name == "mix":
        return _Mixture(mixture or _DEFAULT_MIX, n)
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choices: {', '.join(CHOICES)}")
    return REGISTRY[name](n)
