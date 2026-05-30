"""Mixture sweep — try corpus-weight variants and rank by held-out val (scaled-down AutoMixer, report §3.2.3).

    uv run python scripts/mixture_sweep.py configs/schedules/mixture_sweep.yaml

Runs each variant as a short `train.py --dataset mix --mixture ... --name-suffix <variant>` (sequential,
one GPU) and ranks by the run's final val_loss. NOTE: each run's val is its own mix's 90/10 split, not a
shared eval — so the ranking is indicative. A fixed Design2Code held-out eval is the principled ranker and
is the next step (see docs).
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
    ranking: list[tuple[str, str, float | None]] = []
    for variant in spec["variants"]:
        name, weights = variant["name"], variant["weights"]
        mixture = ",".join(f"{k}={w}" for k, w in weights.items())
        print(f"\n=== variant {name}: {mixture} ===", flush=True)
        subprocess.run(
            [sys.executable, "-m", "laguna_rlvr.visual.train",
             "--dataset", "mix", "--mixture", mixture, "--name-suffix", name,
             "--steps", str(common.get("steps", 800)),
             "--n-train", str(common.get("n_train", 2000)),
             "--pool", str(common.get("pool", 8))],
            check=False,
        )
        res = load_results(experiments_dir="experiments", tag="mm_adapter",
                           config_name=f"{_ENCODER}__{_BASE}__mix__{name}")
        ranking.append((name, mixture, res[-1].get("val_loss") if res else None))

    ranking.sort(key=lambda r: r[2] if r[2] is not None else float("inf"))
    print("\n=== mixture sweep ranking (val_loss — lower is better; in-run val, not a shared eval) ===")
    for name, mixture, val in ranking:
        print(f"  {f'{val:.4f}' if val is not None else 'n/a':>8}  {name}  ({mixture})")


if __name__ == "__main__":
    app()
