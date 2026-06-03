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


def _cauldron(config: str) -> Callable[[int], Dataset]:
    """Builder for a `HuggingFaceM4/the_cauldron` config as an (image, transcription) recon corpus —
    real-image text-rich supervision for Stage-1 (the realism upgrade over generated SyntheticOCR)."""
    def build(n: int) -> Dataset:
        from laguna_rlvr.visual.hf_image_text import CauldronDataset
        return CauldronDataset(config, n=n)
    return build


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
            ds = load_text_image(name, quota)
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

    def text_labels(self) -> list[tuple[str, str]]:
        """(text, corpus) per row WITHOUT decoding images — lets QASFTDataset filter needles cheaply.
        Reads each HF-backed sub-dataset's text column directly (no image decode); only the few non-HF
        corpora (synthetic/swebench) fall back to a per-row read."""
        cols: dict[int, list | None] = {}
        for di, _ in self._index:
            if di not in cols:
                ds = self._datasets[di]
                cols[di] = ds._ds["text"] if hasattr(ds, "_ds") else None
        return [(cols[di][j] if cols[di] is not None else self._datasets[di][j][1], self._names[di])
                for di, j in self._index]


# Default full-training mixture (WebSight-heavy; mirrors the corpus plan in the design doc). A
# hand-set prior — sweep it (scripts/mixture_sweep.py) and pick by held-out val rather than trust it.
# `synthetic` (SyntheticOCR) anchors the GLM-OCR encoder's native text-transcription path: without an
# OCR target the projector — the only trainable bridge — is free to repurpose those dims for the code
# objective and erode readout (multi-turn QA floored at 0 on the code-only mix, 2026-05). kind=None,
# so it's the corpus WER/CER scores against (vs the meaningless code WER it scored before).
_DEFAULT_MIX = [("websight", 0.45), ("webcode2m", 0.25),  # chartmimic dropped: label is a filename,
                ("swebench_mm", 0.1), ("synthetic", 0.2)]  # not code -> no extractable title needle

# Stage-1 projector-ALIGNMENT mix (objective=recon, projector-only): teach the connector to emit
# tokens the frozen LLM will copy TEXT from, before any task tuning — the reference's LLaVA-Pretrain
# step, but text-rich since we're graded on reading (see docs/specs/sft-scale-up-vs-reference.md). The
# transcribe probe (2026-06-02) showed our tokens barely steer the decoder at 2k examples; this is the
# fix, at scale. Reading-dominant (0.8): `synthetic` (SyntheticOCR — generated, free, exact visible-text
# targets, the glyph-copy core) + `cauldron_rendered_text` (REAL rendered-text transcription from
# the_cauldron — the realism hedge over pure synthetic). `websight` is a minority real-image grounding
# slice (recon target is HTML, not reading, so kept small — a code-heavy mix erodes readout, 2026-05).
# Next text-rich adds: pixparse/idl-wds + pdfa-eng-wds (real-doc OCR), the_cauldron iam/textcaps, LLaVAR.
_ALIGN_MIX = [("synthetic", 0.6), ("cauldron_rendered_text", 0.2), ("websight", 0.2)]

REGISTRY: dict[str, Callable[[int], Dataset]] = {
    "synthetic": _synthetic,
    "swebench_mm": _swebench_mm,
    "websight": _websight,
    "webcode2m": _webcode2m,
    "chartmimic": _chartmimic,
    "design2code": _design2code,  # eval-only fixed held-out ranker (don't train on it)
    # the_cauldron text-rich transcription configs — real-image reading supervision for Stage-1.
    "cauldron_rendered_text": _cauldron("rendered_text"),  # rendered text -> transcription
    "cauldron_textcaps": _cauldron("textcaps"),            # reading-aware image captions
    "cauldron_iam": _cauldron("iam"),                      # handwriting transcription
    "cauldron_localized_narratives": _cauldron("localized_narratives"),  # general dense captions (grounding)
    "cauldron_screen2words": _cauldron("screen2words"),                  # UI screenshot -> summary
}
CHOICES = [*REGISTRY, "mix", "align"]

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
        for pat in (_H1_HTML, _TITLE_HTML):  # prefer the VISIBLE <h1> over the browser-tab <title>:
            if m := pat.search(label):       # the <title> ("… :: Stackage Server") often isn't on the page
                if text := re.sub(r"<[^>]+>", "", m.group(1)).strip():  # drop tags nested in <h1>
                    return text
    return None


# The QA read-question per kind — paired with `extract_needle` (it asks for the needle the extractor
# pulls), so it lives beside the taxonomy rather than in the QA harness (cf. TASK_PROMPT).
_READ_Q = {"python": "What is the title of this chart?",
           "html": "What is the title or main heading of this page?"}
_READ_Q_DEFAULT = "What text is shown in this image?"


def read_question(kind: str | None) -> str:
    return _READ_Q.get(kind, _READ_Q_DEFAULT)


class QASFTDataset(Dataset):
    """QA-SFT triples from a base mixture's needle-bearing rows: (image, needle, corpus). The needle
    (chart/page title via `extract_needle`) is the answer to that kind's question, and is NOT in the
    question — so training on it forces the projector to convey vision (vs reconstruction's text-LM
    shortcut). Rows without a clean needle (synthetic / swebench prose) are dropped. Wrap `load_text_image
    ("mix", n)` (its rows carry the corpus tag needed to pick the kind/question).
    """

    def __init__(self, base: Dataset, vqa_sources: list | None = None):
        # Store (source, index) refs and decode the image lazily in __getitem__ — the DataLoader workers
        # then parallelize decode and the Arrow store stays mmap-shared on fork, instead of one process
        # eagerly decoding ~50k PILs into a list (73GB RSS -> OOM at n_train=16000, 2026-06-02). Needles
        # are filtered up front from the text labels (cheap column reads, no image decode).
        self._refs: list[tuple] = []  # (source, idx, corpus, is_vqa, needle)
        if hasattr(base, "text_labels"):
            labels = base.text_labels()
        else:
            labels = [(r[1], r[2] if len(r) > 2 else None) for r in (base[i] for i in range(len(base)))]
        for i, (text, corpus) in enumerate(labels):
            # SyntheticOCR's label IS the visible rendered text -> a clean, diverse, fully-visible reading
            # needle (vs websight's guessable template / webcode2m's off-page <title>); keep it by name
            # here rather than via CORPUS_KIND (which would mis-route its generation-metric token budget).
            needle = (text.strip() or None) if corpus == "synthetic" else extract_needle(text, CORPUS_KIND.get(corpus))
            if needle:
                self._refs.append((base, i, corpus, False, needle))
        for vqa, name in (vqa_sources or []):  # native (image, question, answer) reading supervision
            self._refs += [(vqa, j, f"vqa/{name}", True, None) for j in range(len(vqa))]

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, i: int):  # (image, answer, corpus, question); "" question -> per-kind default
        src, idx, corpus, is_vqa, needle = self._refs[i]
        if is_vqa:
            img, q, ans = src[idx]
            return img, ans, corpus, q
        return src[idx][0], needle, corpus, ""


# VQA reading sets — (image, per-example question, answer) where the answer is short visible text in
# the image: diverse + unguessable, the supervision the title-needle corpora lack. Same shape, so
# adding one is a registry line (a_col may be an answers list -> majority vote, or a single string).
# a_col is a list of annotator answers (majority vote) unless paired=True (q_col/a_col are parallel
# per-image Q&A lists -> first pair). All yield short visible-text answers = clean reading supervision.
VQA_SPECS: dict[str, dict] = {
    "textvqa": dict(repo="lmms-lab/textvqa", split="train", a_col="answers"),                       # scene text
    "chartqa": dict(repo="lmms-lab/ChartQA", split="test", a_col="answer"),                          # charts
    "docvqa": dict(repo="lmms-lab/DocVQA", config="DocVQA", split="validation", a_col="answers"),    # documents
    "ocrvqa": dict(repo="howard-hou/OCR-VQA", q_col="questions", a_col="answers", paired=True),      # cover titles
    "infographicvqa": dict(repo="lmms-lab/DocVQA", config="InfographicVQA", split="validation",      # infographics
                           a_col="answers"),                                                         # (dense text+chart)
}


DEFAULT_VQA = list(VQA_SPECS)  # all registered VQA reading sets — on by default for QA-SFT


# the_cauldron general image-dependent VQA (Stage-2). okvqa excluded: its images are dangling
# /fsx/... path refs (not bundled), so HF's lazy decode hits FileNotFoundError (2026-06-03).
CAULDRON_VQA = ["vqav2", "visual7w"]


def _resolve_vqa(name: str) -> str:
    """Which loader backs a VQA name: 'spec' (lmms-lab via VQA_SPECS) or 'cauldron' (the_cauldron)."""
    if name in VQA_SPECS:
        return "spec"
    if name in CAULDRON_VQA:
        return "cauldron"
    raise ValueError(f"unknown VQA set {name!r}; choices: {list(VQA_SPECS) + CAULDRON_VQA}")


def load_vqa(names: list[str], n: int) -> list[tuple]:
    from laguna_rlvr.visual.hf_image_text import CauldronVQADataset, VQADataset

    def build(name):
        src = _resolve_vqa(name)
        ds = VQADataset(n=n, **VQA_SPECS[name]) if src == "spec" else CauldronVQADataset(name, n=n)
        return ds, name

    # Sequential: each set's own from_generator preload is already process-parallel (LAGUNA_DATASET_PROCS).
    # Do NOT wrap that in a thread pool — launching a multiprocessing pool from many threads breaks HF's
    # from_generator (DatasetGenerationError under parallel preload, 2026-06-03).
    return [build(name) for name in names]


def parse_mixture(spec: str) -> list[tuple[str, float]]:
    """Parse a mixture string 'websight=0.6,webcode2m=0.4' into [(name, weight), ...]."""
    pairs = []
    for part in spec.split(","):
        name, _, weight = part.partition("=")
        pairs.append((name.strip(), float(weight)))
    return pairs


def load_text_image(name: str, n: int, mixture: list[tuple[str, float]] | None = None) -> Dataset:
    if name == "mix":
        return _Mixture(mixture or _DEFAULT_MIX, n)
    if name == "align":  # Stage-1 text-rich projector-alignment mix (override weights via --mixture)
        return _Mixture(mixture or _ALIGN_MIX, n)
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choices: {', '.join(CHOICES)}")
    return REGISTRY[name](n)
