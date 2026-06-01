"""Verifiable multi-turn multimodal QA — does reading + vision-as-tool-observation persist across turns?

Two episode sources:
- **synthetic** (default): toy SyntheticOCR images whose text we know — a fast, offline, in-training
  sanity check (train.py's qa_eval). Measures whether the single-turn projector transfers to multi-turn.
- **mixture**: episodes grounded in the real training corpora (the same screenshots/charts the model
  faces), with verifiability preserved by a needle extracted from each paired label (chart/page title
  via `extract_needle`). Persisted to `data/multiturn_qa.jsonl` so the benchmark is fixed + inspectable;
  images are re-fetched by `build_corpus(corpus)[idx]` (stable row order), so the manifest stays tiny.

Each 3-turn episode reads image A, reads image B, then a text-only follow-up that must recall A's
needle. Scored `qa/metrics/accuracy` (per-turn reading) + `qa/metrics/recall` (cross-turn memory) by
substring (the reply is verbose; substring avoids CER over-penalizing). Baselines share one scoring
core: adapter (vision splice), blind (no images — the floor), tool-mediated (GLM-OCR transcript → chat).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from laguna_rlvr.visual.baseline import _QA_GEN_TOKENS, glm_ocr_transcribe, text_chat
from laguna_rlvr.visual.corpora import CORPUS_KIND, build_corpus, extract_needle, read_question
from laguna_rlvr.visual.data import render_text
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter

_MANIFEST = Path("data/multiturn_qa.jsonl")
_QA_CORPORA = ["webcode2m", "design2code", "websight"]  # html needle-bearing (chartmimic label is a
#                                                         filename, not code -> no extractable needle)
_PER_CORPUS = 64   # rows scanned per corpus when building the manifest
_RECALL_Q = "What was shown in the first image? Answer with its exact title or text."


@dataclass
class QARef:
    """A pointer to one corpus row + its extracted needle (the answer to read/recall)."""
    corpus: str
    idx: int
    needle: str
    kind: str | None


@dataclass
class Episode:
    a: QARef
    b: QARef


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _match(needle: str, reply: str) -> bool:
    """Loosened read match: exact substring OR token-F1 >= 0.5. The strict substring undercounts real
    reads the model phrases differently or partially ('Section 4.2 Results' for 'Section 4.2 Results
    6890', 'University of California' for '…California, Berkeley'); token-F1 credits those without
    rewarding disjoint hallucinations ('nokia' vs 'samsung' -> 0)."""
    n, r = _norm(needle), _norm(reply)
    if n and n in r:
        return True
    nt, rt = set(n.split()), set(r.split())
    if not nt or not rt:
        return False
    inter = len(nt & rt)
    return (2 * inter / (len(nt) + len(rt))) >= 0.5


# ── episodes + manifest ──────────────────────────────────────────────────────────────────────────

def save_manifest(episodes: list[Episode], path: Path = _MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"a": asdict(ep.a), "b": asdict(ep.b)}) for ep in episodes))


def load_manifest(path: Path = _MANIFEST) -> list[Episode]:
    lines = (ln for ln in path.read_text().splitlines() if ln.strip())
    return [Episode(QARef(**d["a"]), QARef(**d["b"])) for d in map(json.loads, lines)]


def synthetic_episodes(n: int = 16, seed: int = 0) -> list[Episode]:
    """Toy episodes from SyntheticOCR — offline sanity; the needle *is* the rendered text."""
    return [Episode(QARef("synthetic", seed + i, f"invoice {seed + i}", None),
                    QARef("synthetic", seed + i + 1000, f"total {seed + 7 * i + 3}", None))
            for i in range(n)]


def mixture_episodes(n: int, corpora: list[str] = _QA_CORPORA, per_corpus: int = _PER_CORPUS) -> list[Episode]:
    """Episodes from the real corpora: scan rows, keep those with a clean needle, pair them up."""
    refs: list[QARef] = []
    for corpus in corpora:
        kind = CORPUS_KIND.get(corpus)
        ds = build_corpus(corpus, per_corpus)
        for idx in range(len(ds)):
            if needle := extract_needle(ds[idx][1], kind):
                refs.append(QARef(corpus, idx, needle, kind))
    return [Episode(refs[2 * i], refs[2 * i + 1]) for i in range(min(n, len(refs) // 2))]


def load_or_build_episodes(source: str, n: int, seed: int, manifest: Path = _MANIFEST) -> list[Episode]:
    """`seed` reshuffles only the synthetic source; the mixture is deterministic (the manifest *is*
    the fixed benchmark — re-read if present, else built once from the corpus scan and persisted)."""
    if source == "synthetic":
        return synthetic_episodes(n, seed)
    if manifest.exists():
        return load_manifest(manifest)[:n]
    episodes = mixture_episodes(n)
    save_manifest(episodes, manifest)
    return episodes


# ── image fetch (cached, re-fetches real rows by index) ────────────────────────────────────────────

def image_fetcher(episodes: list[Episode]):
    """Return `fetch(ref) -> image`, building each real corpus once at `_PER_CORPUS` rows — the same
    size `mixture_episodes` builds at, so the on-disk corpus cache hits instead of re-streaming."""
    cache: dict = {}

    def fetch(ref: QARef):
        if ref.corpus == "synthetic":
            return render_text(ref.needle, seed=ref.idx)
        if ref.corpus not in cache:
            cache[ref.corpus] = build_corpus(ref.corpus, _PER_CORPUS)
        return cache[ref.corpus][ref.idx][0]

    return fetch


# ── per-episode runners (one reply list per episode) ───────────────────────────────────────────────

def _turn_texts(ep: Episode) -> list[str]:
    return [read_question(ep.a.kind), read_question(ep.b.kind), _RECALL_Q]


def adapter_runner(adapter: VisualAdapter, fetch, max_new_tokens: int = _QA_GEN_TOKENS):
    """Vision-splice path: each read turn carries the real image at an <image> marker."""
    def run(ep: Episode) -> list[str]:
        q1, q2, q3 = _turn_texts(ep)
        turns = [Turn(f"{IMAGE_TOKEN}\n{q1}", [fetch(ep.a)]),
                 Turn(f"{IMAGE_TOKEN}\n{q2}", [fetch(ep.b)]),
                 Turn(q3)]
        return adapter.chat(turns, max_new_tokens=max_new_tokens)
    return run


def blind_runner(llm, tok, max_new_tokens: int = _QA_GEN_TOKENS):
    """Floor: text-only chat, no images — how much is answerable without seeing anything."""
    def run(ep: Episode) -> list[str]:
        return text_chat(llm, tok, _turn_texts(ep), max_new_tokens=max_new_tokens)
    return run


def ocr_runner(llm, tok, transcripts: dict[tuple[str, int], str], max_new_tokens: int = _QA_GEN_TOKENS):
    """Tool-mediated: each read turn carries the image's GLM-OCR transcript (pre-computed, keyed by
    (corpus, idx) so the staged-GPU harness transcribes every image once before the base LLM loads)."""
    def run(ep: Episode) -> list[str]:
        q1, q2, q3 = _turn_texts(ep)
        ta, tb = transcripts[(ep.a.corpus, ep.a.idx)], transcripts[(ep.b.corpus, ep.b.idx)]
        return text_chat(llm, tok, [f"{ta}\n{q1}", f"{tb}\n{q2}", q3], max_new_tokens=max_new_tokens)
    return run


def transcribe_episodes(episodes: list[Episode], fetch, device: str = "cuda") -> dict[tuple[str, int], str]:
    """GLM-OCR transcript per distinct image across all episodes (deduped), keyed by (corpus, idx)."""
    refs = {(r.corpus, r.idx): r for ep in episodes for r in (ep.a, ep.b)}
    texts = glm_ocr_transcribe([(fetch(r),) for r in refs.values()], device=device)
    return dict(zip(refs, texts))


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────

def run_qa(run_episode, episodes: list[Episode], prefix: str = "qa") -> dict[str, float]:
    """Aggregate per-turn reading accuracy + cross-turn recall over episodes (substring match).
    `prefix` namespaces the keys (`qa` for the adapter; `qa/<baseline>` for the baseline panel)."""
    hits = total = recall = 0
    for ep in episodes:
        r1, r2, r3 = run_episode(ep)
        for ref, reply in ((ep.a, r1), (ep.b, r2)):
            total += 1
            hits += _match(ref.needle, reply)
        recall += _match(ep.a.needle, r3)
    return {f"{prefix}/metrics/accuracy": hits / max(total, 1),
            f"{prefix}/metrics/recall": recall / max(len(episodes), 1)}


def dataset_qa_accuracy(adapter: VisualAdapter, items: list, max_new_tokens: int = _QA_GEN_TOKENS,
                        prefix: str = "qa") -> dict[str, float]:
    """Single-turn read accuracy over QASFTDataset val items (image, answer, corpus, question) — scores
    the ACTUAL training distribution (incl. webcode2m visible-H1 / SyntheticOCR / the VQA suite), broken
    down per corpus, rather than a fixed websight-heavy manifest. Substring match (replies are verbose)."""
    from collections import defaultdict

    per_corpus: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for img, answer, corpus, question in items:
        q = question or read_question(CORPUS_KIND.get(corpus))
        reply = adapter.chat([Turn(f"{IMAGE_TOKEN}\n{q}", [img])], max_new_tokens=max_new_tokens)[0]
        per_corpus[corpus][0] += int(_match(answer, reply))
        per_corpus[corpus][1] += 1
    hits, total = (sum(c) for c in zip(*per_corpus.values())) if per_corpus else (0, 0)
    out = {f"{prefix}/metrics/accuracy": hits / max(total, 1)}
    out.update({f"{prefix}/metrics/acc_{c.replace('/', '_')}": h / max(n, 1)
                for c, (h, n) in per_corpus.items()})
    return out


def evaluate_multiturn_qa(adapter: VisualAdapter, n: int = 16, seed: int = 0,
                          max_new_tokens: int = _QA_GEN_TOKENS, *, source: str = "synthetic",
                          manifest: Path = _MANIFEST, prefix: str = "qa") -> dict[str, float]:
    """Adapter multi-turn QA — per-turn reading accuracy + cross-turn recall (does it still recall image
    A's needle after reading B + a text turn = conversation memory). `source="mixture"` = real-corpora
    3-turn episodes. `prefix` namespaces the keys (use 'qa_mt' so it doesn't collide with single-turn)."""
    episodes = load_or_build_episodes(source, n, seed, manifest)
    fetch = image_fetcher(episodes)
    return run_qa(adapter_runner(adapter, fetch, max_new_tokens), episodes, prefix=prefix)
