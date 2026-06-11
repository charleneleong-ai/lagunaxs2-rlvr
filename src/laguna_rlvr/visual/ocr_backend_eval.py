"""OCR-backend bake-off by extraction quality — which backend should feed the agentic loop?

The real-OCR loop (docs/experiments/agentic-ocr-tool/real_ocr_loop.md) showed loop success tracks whether
the OCR backend put the answer in the transcript: GLM-OCR's per-corpus extractability (visualmrc 0.04 …
textvqa 0.60) is the CEILING on loop success. This bake-off measures that ceiling directly, decoupled from
the decoder, on two complementary axes:

  answer-coverage   7 glyph VQA corpora        fraction of items whose gold answer is recoverable from the
                                               transcript (the loop's own `_match`) — reference-free, and
                                               exactly the per-corpus ceiling on loop success.
  WER / CER         cauldron_rendered_text     literal word/char error rate vs the rendered-text reference
                                               (the label IS the visible text) — decoder-independent
                                               extraction quality on a reference-bearing set.

Backends: `glm_ocr` (zai-org/GLM-OCR, OCR-native) vs `qwen3_vl` (Qwen3-VL-8B, general VLM). Each backend
transcribes in its OWN process (memory isolation — the two multi-GB VLMs never co-reside), caching to disk;
the parent reads caches and scores on CPU.

    uv run python -m laguna_rlvr.visual.ocr_backend_eval transcribe --backend qwen3_vl --probe glyph --n 40
    uv run python -m laguna_rlvr.visual.ocr_backend_eval bakeoff --n 40
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import typer

from laguna_rlvr.visual.baseline import vlm_transcribe
from laguna_rlvr.visual.corpora import load_text_image
from laguna_rlvr.visual.encoders import _REPOS
from laguna_rlvr.visual.metrics import cer as _cer, wer as _wer
from laguna_rlvr.visual.multiturn_qa import _match, _norm
from laguna_rlvr.visual.tool_eval import GLYPH_VQA, load_items

# the general VLM is the 8B (the stronger general tower from the encoder bake-off, PR #33)
BACKENDS = {"glm_ocr": _REPOS["glm_ocr"], "qwen3_vl": _REPOS["qwen3_vl_8b"]}
REF_CORPUS = "cauldron_rendered_text"   # label == the visible rendered text -> a real WER reference
_OUT = Path("results/ocr_backend")


def wer(ref: str, hyp: str) -> float:
    """Word error rate vs reference. Normalize (lower + collapse whitespace) first — the repo's jiwer
    metric is case-sensitive, so 'NOKIA' vs 'nokia' would otherwise score a spurious 1.0."""
    return _wer(_norm(hyp), _norm(ref))   # metrics order is (pred, ref)


def cer(ref: str, hyp: str) -> float:
    """Character error rate vs reference, with the same normalization."""
    return _cer(_norm(hyp), _norm(ref))


def coverage_matrix(rows: list[dict]) -> dict[str, float]:
    """rows = [{corpus, gold, transcript}] -> per-corpus answer-coverage + an 'overall' micro-average.
    Coverage of an item = the loop's `_match(gold, transcript)`: is the answer recoverable from the OCR?"""
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        cell = hits[r["corpus"]]
        cell[0] += int(_match(r["gold"], r["transcript"]))
        cell[1] += 1
    out = {c: h / n for c, (h, n) in hits.items()}
    tot_h, tot_n = (sum(v) for v in zip(*hits.values())) if hits else (0, 0)
    return {"overall": tot_h / max(tot_n, 1), **out}


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def _cache_path(backend: str, probe: str, n: int) -> Path:
    return _OUT / f"{backend}__{probe}__n{n}.jsonl"


def _glyph_corpora() -> list[str]:
    return _csv(GLYPH_VQA)


def _transcribe(backend: str, probe: str, n: int, out: Path, device: str) -> None:
    """Transcribe one probe set with one backend, writing aligned {…, transcript} rows. `glyph` carries
    (corpus, gold) for coverage; `ref` carries the rendered-text reference for WER/CER."""
    repo = BACKENDS[backend]
    if probe == "glyph":
        items = load_items(_glyph_corpora(), n)
        texts = vlm_transcribe([(img,) for _, img, _, _ in items], repo, device=device)
        rows = [{"corpus": c, "gold": str(gold), "transcript": t}
                for (c, _img, _q, gold), t in zip(items, texts)]
    else:
        ds = load_text_image(REF_CORPUS, n)
        items = [ds[i] for i in range(min(n, len(ds)))]
        texts = vlm_transcribe([(img,) for img, *_ in items], repo, device=device)
        rows = [{"ref": ref, "transcript": t} for (_img, ref, *_), t in zip(items, texts)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[transcribe] {backend}/{probe}: wrote {len(rows)} rows -> {out}", flush=True)


def _ensure(backend: str, probe: str, n: int, device: str) -> list[dict]:
    """Cache transcripts per (backend, probe, n) in a CHILD process so the two VLMs never co-reside; the
    parent only ever reads the cache + scores on CPU. Returns the cached rows."""
    path = _cache_path(backend, probe, n)
    if not path.exists():
        subprocess.run([sys.executable, "-u", "-m", "laguna_rlvr.visual.ocr_backend_eval", "transcribe",
                        "--backend", backend, "--probe", probe, "--n", str(n),
                        "--out", str(path), "--device", device], check=True)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _render_coverage(cov: dict[str, dict[str, float]], corpora: list[str], backends: list[str]) -> str:
    head = f"| {'corpus':<16} | " + " | ".join(f"{b:<10}" for b in backends) + " |"
    sep = "|" + "-" * 18 + "|" + "|".join("-" * 12 for _ in backends) + "|"
    rows = [head, sep]
    for c in ["overall", *corpora]:
        cells = " | ".join(f"{cov[b].get(c, float('nan')):<10.2f}" for b in backends)
        rows.append(f"| {c:<16} | {cells} |")
    return "\n".join(rows)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def transcribe(
    backend: str = typer.Option(..., help=f"OCR backend: {list(BACKENDS)}"),
    probe: str = typer.Option(..., help="'glyph' (coverage) or 'ref' (WER/CER reference corpus)"),
    n: int = typer.Option(40, help="items per corpus"),
    out: str = typer.Option("", help="cache path (default: derived from backend/probe/n)"),
    device: str = typer.Option("cuda"),
) -> None:
    """Transcribe one probe set with one backend, in its OWN process (so the VLM's memory is reclaimed
    on exit and the two backends never co-reside)."""
    if backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend {backend!r}; choices: {list(BACKENDS)}")
    _transcribe(backend, probe, n, Path(out) if out else _cache_path(backend, probe, n), device)


@app.command()
def bakeoff(
    backends: str = typer.Option(",".join(BACKENDS), help="comma list of OCR backends"),
    n: int = typer.Option(40, help="items per corpus"),
    out: str = typer.Option("results/ocr_backend/bakeoff.json", help="where to write the result"),
    device: str = typer.Option("cuda"),
) -> None:
    """Compare OCR backends on answer-coverage (7 glyph corpora) + WER/CER (rendered-text reference).
    Auto-transcribes each (backend, probe) in its own process, then scores on CPU."""
    names = _csv(backends)
    cov: dict[str, dict[str, float]] = {}
    err: dict[str, dict[str, float]] = {}
    for b in names:
        cov[b] = coverage_matrix(_ensure(b, "glyph", n, device))
        ref_rows = _ensure(b, "ref", n, device)
        err[b] = {"wer": sum(wer(r["ref"], r["transcript"]) for r in ref_rows) / max(len(ref_rows), 1),
                  "cer": sum(cer(r["ref"], r["transcript"]) for r in ref_rows) / max(len(ref_rows), 1)}

    corpora = _glyph_corpora()
    table = _render_coverage(cov, corpora, names)
    err_table = "\n".join(f"  {b:<10} WER {err[b]['wer']:.3f}  CER {err[b]['cer']:.3f}" for b in names)
    print(f"\nAnswer-coverage (loop-success ceiling):\n{table}\n\n"
          f"Transcription error vs {REF_CORPUS} reference:\n{err_table}", flush=True)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n": n, "backends": names, "coverage": cov, "error": err}, indent=2))
    print(f"\nwrote {out_path}", flush=True)


@app.command("build-docs")
def build_docs(
    backend: str = typer.Option("qwen3_vl", help=f"OCR backend whose transcript feeds the loop: {list(BACKENDS)}"),
    n: int = typer.Option(40, help="items per corpus"),
    out: str = typer.Option("results/ocr_backend/loop_docs.jsonl", help="docs pack for conf/env/ocr_tool_real"),
    device: str = typer.Option("cuda"),
) -> None:
    """Emit the agentic loop's {cat,id,text,q,a} docs pack with `backend`'s transcript as `text`, so the
    loop's ocr() serves exactly the transcripts this bake-off scored. Reuses the per-backend glyph cache
    (no GPU when it exists), zipping it with each item's question/gold. Default `qwen3_vl` — the verdict
    backend (beats GLM-OCR on WER + coverage; see docs/experiments/agentic-ocr-tool/ocr_backend_wer.md)."""
    rows = _ensure(backend, "glyph", n, device)   # {corpus, gold, transcript}, aligned to load_items(glyph)
    items = load_items(_glyph_corpora(), n)
    assert len(rows) == len(items), f"{len(rows)} cache rows vs {len(items)} items"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, (r, (corpus, _img, q, gold)) in enumerate(zip(rows, items)):
            f.write(json.dumps({"cat": corpus, "id": f"{corpus}_{i}.png",
                                "text": r["transcript"], "q": q, "a": str(gold)}) + "\n")
    print(f"[build-docs] {backend}: wrote {len(items)} docs -> {out_path}", flush=True)


if __name__ == "__main__":
    app()
