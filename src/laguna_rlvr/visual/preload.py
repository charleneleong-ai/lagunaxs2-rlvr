"""Preload real-image corpora into the on-disk cache, GPU-free and decoupled from training.

The streaming + save_to_disk for large real-image counts (cauldron, websight) is slow and stalled the
training process twice — burning A100 time while the GPU sat idle. But `_cached_or_stream` keys the
cache by (corpus, count, ...) and skips streaming once the dir exists, so preloading it ONCE here
(no model loaded, no GPU) makes every later training run load instantly. Run the target counts the
Stage-1/Stage-2 mixes will request (the cache is count-keyed, so the count must match).

    laguna-preload cauldron_rendered_text:20000 websight:12000
"""
from __future__ import annotations

import time

import typer

from laguna_rlvr.visual.corpora import build_corpus

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(specs: list[str] = typer.Argument(..., help="corpus:count, e.g. cauldron_rendered_text:20000")) -> None:
    """Build each `corpus:count` so its on-disk cache is populated (no GPU, no model)."""
    for spec in specs:
        name, _, count = spec.partition(":")
        n = int(count)
        t = time.perf_counter()
        print(f"[preload] {name} ({n}) — streaming + save_to_disk ...", flush=True)
        ds = build_corpus(name, n)
        print(f"[preload] {name}: cached {len(ds)} rows in {time.perf_counter() - t:.0f}s", flush=True)
    print("[preload] done", flush=True)


if __name__ == "__main__":
    app()
