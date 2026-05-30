"""Mixture sweep — try corpus-weight variants and rank by held-out val (scaled-down AutoMixer, report §3.2.3).

    uv run python scripts/mixture_sweep.py configs/schedules/mixture_sweep.yaml

Runs each variant as a short `train.py --dataset mix --mixture ... --eval-dataset <fixed>` (sequential,
one GPU) and ranks by eval_loss on the FIXED held-out eval (default design2code) — the same unseen set
for every variant, so the ranking is attributable to the mixture (AutoMixer §3.2.3). val_loss (in-mix
90/10) is also logged per run, for diagnosing under-training vs genuinely-worse.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
import yaml
from autoresearch.results import load_results

app = typer.Typer(add_completion=False)
_ENCODER, _BASE = "glm_ocr", "Laguna-XS.2"  # the default setup this sweep targets (train.py run_name)


@app.command()
def main(schedule: Path) -> None:
    spec = yaml.safe_load(schedule.read_text())
    common = spec.get("common_overrides", {})
    eval_dataset = common.get("eval_dataset", "design2code")
    ranking: list[tuple[str, str, float | None, float | None]] = []
    for variant in spec["variants"]:
        name, weights = variant["name"], variant["weights"]
        mixture = ",".join(f"{k}={w}" for k, w in weights.items())
        print(f"\n=== variant {name}: {mixture} ===", flush=True)
        subprocess.run(
            [sys.executable, "-m", "laguna_rlvr.visual.train",
             "--dataset", "mix", "--mixture", mixture, "--name-suffix", name,
             "--eval-dataset", eval_dataset,
             "--steps", str(common.get("steps", 800)),
             "--n-train", str(common.get("n_train", 2000)),
             "--pool", str(common.get("pool", 8))],
            check=False,
        )
        res = load_results(experiments_dir="experiments", tag="mm_adapter",
                           config_name=f"{_ENCODER}__{_BASE}__mix__{name}")
        row = res[-1] if res else {}
        ranking.append((name, mixture, row.get("eval_loss"), row.get("val_loss")))

    ranking.sort(key=lambda r: r[2] if r[2] is not None else float("inf"))
    print(f"\n=== mixture sweep ranking by eval/loss on '{eval_dataset}' (lower is better) ===")
    print(f"  {'eval':>8}  {'val':>8}  variant")
    for name, mixture, ev, val in ranking:
        ev_s = f"{ev:.4f}" if ev is not None else "n/a"
        val_s = f"{val:.4f}" if val is not None else "n/a"
        print(f"  {ev_s:>8}  {val_s:>8}  {name}  ({mixture})")


if __name__ == "__main__":
    app()
