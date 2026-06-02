"""Preload real-image corpora into the on-disk cache, GPU-free and decoupled from training.

The streaming + save_to_disk for large real-image counts (cauldron, websight) is slow and stalled the
training process twice — burning A100 time while the GPU sat idle. But `_cached_or_stream` keys the
cache by (corpus, count, ...) and skips streaming once the dir exists, so preloading it ONCE here
(no model loaded, no GPU) makes every later training run load instantly. Run the target counts the
Stage-1/Stage-2 mixes will request (the cache is count-keyed, so the count must match).

    laguna-preload cauldron_rendered_text:20000 websight:12000 textvqa:16000
"""
from __future__ import annotations

import os
import time

import typer

from laguna_rlvr.visual.corpora import VQA_SPECS, build_corpus, load_vqa

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(specs: list[str] = typer.Argument(..., help="corpus-or-vqa:count, e.g. cauldron_rendered_text:20000 textvqa:16000"),
         procs: int = typer.Option(1, help="parallel encode workers (file-sharded). The Arrow image "
                                   "encode is ~0.3s/img single-core; set to ~num_cores for ~Nx faster.")) -> None:
    """Populate the on-disk cache for each `name:count` (no GPU, no model). `name` is a build_corpus
    corpus OR a VQA reading set (VQA_SPECS), so the Stage-2 VQA suite can be preloaded too."""
    os.environ["LAGUNA_DATASET_PROCS"] = str(procs)  # read at stream-time by the loaders' _shard_plan
    for spec in specs:
        name, _, count = spec.partition(":")
        n = int(count)
        t = time.perf_counter()
        print(f"[preload] {name} ({n}) — streaming + save_to_disk ...", flush=True)
        ds = load_vqa([name], n)[0][0] if name in VQA_SPECS else build_corpus(name, n)
        print(f"[preload] {name}: cached {len(ds)} rows in {time.perf_counter() - t:.0f}s", flush=True)
    print("[preload] done", flush=True)


if __name__ == "__main__":
    app()
