"""Stream an HF (image, text) dataset into (screenshot, code) pairs, cached to disk after first fetch.

For corpora whose screenshot is an embedded `Image` column — WebSight, WebCode2M, ChartMimic — so there
is no URL download (unlike SWE-bench M). These sets are huge (WebSight 31GB+, WebCode2M ~1TB), so we
`streaming=True` and materialize only the first `n` rows rather than downloading the whole parquet.

The streamed rows are then saved to a compressed local cache (`datasets.save_to_disk`) keyed by the load
params, so every later run / restart loads from disk — no network, no mid-stream `ReadTimeout`. Point
`LAGUNA_DATA_CACHE` elsewhere to relocate it (default `~/.cache/laguna-mm/corpora`).

The label is the page's HTML/code, truncated to keep projector-SFT sequences bounded; the full
screenshot->code objective with untruncated code belongs to the later long-context / RLVR stage.
"""
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from itertools import islice
from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import Features
from datasets import Image as HFImage
from datasets import Value, load_dataset, load_from_disk
from rich.progress import track
from torch.utils.data import Dataset

_CACHE_DIR = Path(os.environ.get("LAGUNA_DATA_CACHE", Path.home() / ".cache" / "laguna-mm" / "corpora"))


def _cached_or_stream(key: str, stream_fn: Callable[[], HFDataset]) -> HFDataset:
    """Disk-cached materialization: load the streamed rows from `_CACHE_DIR/key`, or run `stream_fn`
    once and cache it. Shared by the dataset classes so the cache contract lives in one place."""
    cache = _CACHE_DIR / key.replace("/", "_")
    if cache.exists():
        return load_from_disk(str(cache))
    ds = stream_fn()
    cache.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache))
    return ds


_IMG_TEXT_FEATURES = Features({"image": HFImage(), "text": Value("string")})


def _image_text_rows(*, repo, config, split, n, offset, image_col, text_col, max_text_chars):
    """Yield {image, text} rows from a flat HF (image, text) dataset — for from_generator (incremental,
    low-memory). gen_kwargs carry the args so from_generator fingerprints the cache by them."""
    stream = load_dataset(repo, config, split=split, streaming=True)
    kept = 0
    for row in track(islice(stream, offset, offset + n), total=n, description=f"{repo} ({n})"):
        img, txt = row.get(image_col), row.get(text_col)
        if img is not None and txt:
            kept += 1
            yield {"image": img.convert("RGB"), "text": txt[:max_text_chars]}
    if kept == 0:
        raise RuntimeError(f"no usable rows from {repo} (cols {image_col!r}/{text_col!r})")


def _cauldron_rows(*, config, split, n, offset, max_text_chars):
    """Yield {image, transcription} rows from a the_cauldron config — for from_generator."""
    stream = load_dataset("HuggingFaceM4/the_cauldron", config, split=split, streaming=True)
    kept = 0
    for row in track(islice(stream, offset, offset + n), total=n, description=f"cauldron/{config} ({n})"):
        images, texts = row.get("images"), row.get("texts")
        if images and texts and texts[0].get("assistant"):
            kept += 1
            yield {"image": images[0].convert("RGB"), "text": texts[0]["assistant"][:max_text_chars]}
    if kept == 0:
        raise RuntimeError(f"no usable rows from the_cauldron/{config}")


class HFImageTextDataset(Dataset):
    """(screenshot, code) pairs streamed from an HF dataset (embedded image + text), cached to disk."""

    def __init__(self, repo: str, *, config: str | None = None, split: str = "train",
                 n: int = 2000, offset: int = 0, image_col: str = "image", text_col: str = "text",
                 max_text_chars: int = 2048):
        key = "__".join(str(p) for p in (repo, config, split, n, offset, image_col, text_col, max_text_chars))
        # in-memory Arrow; kept lazy — decode per access, not all n images up front
        self._ds = _cached_or_stream(
            key, lambda: self._stream(repo, config, split, n, offset, image_col, text_col, max_text_chars))

    @staticmethod
    def _stream(repo: str, config: str | None, split: str, n: int, offset: int,
                image_col: str, text_col: str, max_text_chars: int) -> HFDataset:
        # from_generator writes Arrow shards incrementally as rows are yielded (one image in flight) —
        # low memory + scalable. from_dict held all N decoded images in RAM and encoded them in one
        # monolithic pass: ~24GB + single-shot encode at N=8000 stalled training/preload (2026-06-02).
        # offset skips the first `offset` rows -> a held-out eval slice disjoint from the training range.
        return HFDataset.from_generator(
            _image_text_rows, features=_IMG_TEXT_FEATURES,
            gen_kwargs=dict(repo=repo, config=config, split=split, n=n, offset=offset,
                            image_col=image_col, text_col=text_col, max_text_chars=max_text_chars))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        row = self._ds[i]
        return row["image"], row["text"]


class CauldronDataset(Dataset):
    """(image, transcription) pairs from a `HuggingFaceM4/the_cauldron` config — a curated 50-set VLM
    collection. Rows are `{images: [PIL], texts: [{user, assistant, source}, ...]}`; we take the first
    image and the first turn's `assistant` as the recon target. Text-rich transcription/caption configs
    (rendered_text, iam, textcaps) give REAL-image reading supervision for Stage-1 alignment — the
    realism upgrade over SyntheticOCR. Same streaming + disk cache as HFImageTextDataset."""

    def __init__(self, config: str, *, split: str = "train", n: int = 2000, offset: int = 0,
                 max_text_chars: int = 2048):
        key = "cauldron__" + "__".join(str(p) for p in (config, split, n, offset, max_text_chars))
        self._ds = _cached_or_stream(key, lambda: self._stream(config, split, n, offset, max_text_chars))

    @staticmethod
    def _stream(config, split, n, offset, max_text_chars) -> HFDataset:
        # incremental write (see HFImageTextDataset._stream) — from_dict on N large images stalled
        return HFDataset.from_generator(
            _cauldron_rows, features=_IMG_TEXT_FEATURES,
            gen_kwargs=dict(config=config, split=split, n=n, offset=offset, max_text_chars=max_text_chars))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        row = self._ds[i]
        return row["image"], row["text"]


class VQADataset(Dataset):
    """(image, question, answer) triples from a VQA set (TextVQA/DocVQA): embedded image + a
    per-example question + a list of annotator answers (majority vote). The answer is short, diverse
    visible text in the image — the clean, unguessable reading supervision our title-needle corpora
    lack. Same streaming + disk cache as HFImageTextDataset."""

    def __init__(self, repo: str, *, config: str | None = None, split: str = "train", n: int = 2000,
                 offset: int = 0, image_col: str = "image", q_col: str = "question", a_col: str = "answers",
                 paired: bool = False):
        key = "vqa__" + "__".join(str(p) for p in (repo, config, split, n, offset, paired))
        self._ds = _cached_or_stream(
            key, lambda: self._stream(repo, config, split, n, offset, image_col, q_col, a_col, paired))

    @staticmethod
    def _stream(repo, config, split, n, offset, image_col, q_col, a_col, paired) -> HFDataset:
        stream = load_dataset(repo, config, split=split, streaming=True)
        imgs, qs, ans = [], [], []
        for row in track(islice(stream, offset, offset + n), total=n, description=f"{repo} ({n})"):
            img, q, a = row.get(image_col), row.get(q_col), row.get(a_col)
            if paired:  # q_col/a_col are PARALLEL lists (multiple Q&A per image, e.g. OCR-VQA): first pair
                q = q[0] if isinstance(q, list) and q else q
                a = a[0] if isinstance(a, list) and a else a
            else:  # a_col is a list of annotator answers to the single question -> majority vote
                a = Counter(a).most_common(1)[0][0] if isinstance(a, list) and a else a
            if img is not None and q and a:
                imgs.append(img.convert("RGB"))
                qs.append(q)
                ans.append(a)
        if not imgs:
            raise RuntimeError(f"no usable rows from {repo} (cols {image_col!r}/{q_col!r}/{a_col!r})")
        return HFDataset.from_dict(
            {"image": imgs, "question": qs, "answer": ans},
            features=Features({"image": HFImage(), "question": Value("string"), "answer": Value("string")}))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        row = self._ds[i]
        return row["image"], row["question"], row["answer"]


class ScreenSpotDataset(Dataset):
    """SCAFFOLD — GUI grounding (ScreenSpot: image + instruction + normalized bbox). Framed as a
    read+localize triple: (image, "locate '{instruction}' …", bbox-as-text). The whole point of vision
    on a coding/agentic model. NOTE: grounding is a *different objective* than reading — the answer is
    a box, so it should be scored by IoU, not the substring/token-F1 reading metric, and it is NOT in
    DEFAULT_VQA yet. Wire it deliberately once box-IoU scoring is added."""

    def __init__(self, repo: str = "rootsautomation/ScreenSpot", *, split: str = "test", n: int = 2000,
                 offset: int = 0):
        key = "screenspot__" + "__".join(str(p) for p in (repo, split, n, offset))
        self._ds = _cached_or_stream(key, lambda: self._stream(repo, split, n, offset))

    @staticmethod
    def _stream(repo, split, n, offset) -> HFDataset:
        stream = load_dataset(repo, split=split, streaming=True)
        imgs, qs, ans = [], [], []
        for row in track(islice(stream, offset, offset + n), total=n, description=f"{repo} ({n})"):
            img, instr, box = row.get("image"), row.get("instruction"), row.get("bbox")
            if img is not None and instr and box and len(box) == 4:
                imgs.append(img.convert("RGB"))
                qs.append(f"Locate the UI element for '{instr}'. Give its bounding box as [x1, y1, x2, y2] in 0-1.")
                ans.append("[" + ", ".join(f"{c:.3f}" for c in box) + "]")
        if not imgs:
            raise RuntimeError(f"no usable rows from {repo}")
        return HFDataset.from_dict(
            {"image": imgs, "question": qs, "answer": ans},
            features=Features({"image": HFImage(), "question": Value("string"), "answer": Value("string")}))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        row = self._ds[i]
        return row["image"], row["question"], row["answer"]
