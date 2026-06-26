"""ocrvqa-recovery sweep — train the baseline + knob variants from a schedule yaml, then probe + rank.

    uv run python scripts/ocrvqa_recovery_sweep.py configs/schedules/ocrvqa_recovery.yaml

Mirrors mixture_sweep.py: reads `common_overrides` + `variants` (each `{suffix, flags}`) from the schedule,
runs each as a sequential one-GPU `vision_tool_gspo` subprocess (only the named knob differs), then probes
every checkpoint on the 7-corpus glyph set and ranks by overall solve-rate with an OCRVQA FLOOR guard —
the verdict is `overall >= 0.171 AND ocrvqa >= 0.20` (a net win must not come at ocrvqa's expense). The yaml
is the single source of truth; pass `--steps` to override the schedule for a faster directional screen.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False)
_OVERALL_FLOOR, _OCRVQA_FLOOR = 0.171, 0.20  # the merged GSPO baseline overall; ocrvqa recovery target


def _common_args(common: dict, steps: int | None) -> list[str]:
    return ["--init-ckpt", common["init_ckpt"], "--base", common["base"],
            "--steps", str(steps or common["steps"]), "--lr", str(common["lr"]),
            "--group-size", str(common["group_size"]), "--batch", str(common["batch"]),
            "--n-train", str(common["n_train"]), "--eval-n", str(common["eval_n"]),
            "--difficulty" if common.get("difficulty", True) else "--uniform",
            "--wandb" if common.get("wandb", False) else "--no-wandb"]


def _solve_rates(slug: str) -> tuple[float, float]:
    """(overall, ocrvqa) mean solve-rate from the probe record, or (nan, nan) if it's missing."""
    rec = Path(f"results/probe/vision_tool__{slug}.jsonl")
    if not rec.exists():
        return float("nan"), float("nan")
    by: dict[str, list[bool]] = defaultdict(list)
    for line in rec.read_text().splitlines():
        r = json.loads(line)
        by[r["corpus"]].append(bool(r["success"]))
    allv = [s for v in by.values() for s in v]
    ocr = by.get("ocrvqa", [])
    return (sum(allv) / len(allv), sum(ocr) / len(ocr) if ocr else float("nan"))


@app.command()
def main(schedule: Path,
         steps: int = typer.Option(0, help="override the schedule's steps for a faster directional screen")) -> None:
    spec = yaml.safe_load(schedule.read_text())
    common, variants = spec["common_overrides"], spec["variants"]
    base_args = _common_args(common, steps or None)
    base_name = Path(common["base"]).name

    for v in variants:
        print(f"\n=== TRAIN {v['name']} ({v.get('flags') or 'baseline'}) ===", flush=True)
        subprocess.run([sys.executable, "-m", "laguna_rlvr.visual.vision_tool_gspo",
                        *base_args, "--name-suffix", v["suffix"], *shlex.split(v.get("flags", ""))], check=False)

    for v in variants:
        ckpt = f"results/visual/glm_ocr__{base_name}__{v['suffix']}/best.pt"
        print(f"\n=== PROBE {v['name']} ({ckpt}) ===", flush=True)
        subprocess.run([sys.executable, "-m", "laguna_rlvr.visual.vision_tool_eval", "probe", ckpt,
                        "--base", common["base"], "--slug", v["suffix"]], check=False)

    print(f"\n=== ocrvqa-recovery ranking (verdict = overall>={_OVERALL_FLOOR} AND ocrvqa>={_OCRVQA_FLOOR}) ===")
    print(f"  {'overall':>8}  {'ocrvqa':>8}  verdict  variant")
    rows = [(v["name"], *_solve_rates(v["suffix"])) for v in variants]
    for name, overall, ocrvqa in sorted(rows, key=lambda r: (-r[1], -(r[2] if r[2] == r[2] else -1))):
        ok = overall >= _OVERALL_FLOOR and ocrvqa >= _OCRVQA_FLOOR
        print(f"  {overall:8.3f}  {ocrvqa:8.3f}  {'PASS ' if ok else '  -  '}   {name}")


if __name__ == "__main__":
    app()
