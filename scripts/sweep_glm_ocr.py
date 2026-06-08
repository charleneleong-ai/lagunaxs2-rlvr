"""GLM-OCR Stage-1 -> (gated) Stage-2 pipeline driven by autoresearch's SweepRunner, so each train.py
iter runs under ACTIVE GPU triage (auto-kill hangs / wasted-compute / undersized configs -> EARLY_KILL)
with orphan-process cleanup between iters — instead of an ad-hoc bash retry loop.

    uv run python scripts/sweep_glm_ocr.py            # waits for the GPU, then runs the pipeline

Why this over the bash daemon: train.py only does PASSIVE GPU monitoring (records peak_mem/util into
results.jsonl). The active auto-fixing — killing a run that hangs at <8% util for 5min, wastes compute at
<25% for 20min, or is undersized — lives at the SweepRunner level via GPUTriageMonitor, and the janitor
reaps any PPID=1 DataLoader-worker orphans a kill leaves behind before the next iter binds the GPU.

The planner is feedback-gated: Stage-2 only runs if Stage-1 produced a grounded checkpoint (best.pt).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from autoresearch.gpu_monitor import GPUTriageThresholds
from autoresearch.results import load_results
from autoresearch.sweep_runner import GPUTriageMonitor, IterPlan, SweepRunner

REPO = Path("/home/ubuntu/lagunaxs2-rlvr")
CFG = "configs/mm_adapter/a100-80gb-laguna-bf16-microbatch2.toml"
BASE = "poolside/Laguna-XS.2"
TAG = "mm_adapter"  # experiments/<TAG>/<config_name>/results.jsonl (train.py's layout)
INSTRUCT_MIX = "websight=0.4,webcode2m=0.3,synthetic=0.3"
DOC_VQA = "docvqa,ocrvqa,infographic_vqa,visualmrc,vqav2"  # glyph outcomes + vqav2 grounding control
STAGE1_CKPT = REPO / "results/visual/glm_ocr__Laguna-XS.2__align__stage1caption/best.pt"


def _cfg_name(dataset: str, suffix: str) -> str:
    # mirrors train.py's run dir: {encoder}__{base_name}__{dataset}__{suffix}, lower-cased on disk
    return f"glm_ocr__Laguna-XS.2__{dataset}__{suffix}".lower()


def _train(*args: str, pool: str = "1") -> list[str]:
    return [sys.executable, "-u", "-m", "laguna_rlvr.visual.train",
            "--config", CFG, "--base", BASE, "--encoder", "glm_ocr",
            "--projector", "resampler", "--anchor", "--lr", "2e-5",
            "--n-queries", "256", "--pool", pool, *args]


def _stage1() -> IterPlan:
    return IterPlan(
        cmd=_train("--dataset", "align", "--name-suffix", "stage1caption",
                   "--steps", "3000", "--n-train", "200000",
                   "--description", "GLM-OCR Stage-1 caption alignment (projector-only, recon LM loss); "
                   "grounds the resampler in GLM-OCR's OCR-native feature space (d_enc=1536)."),
        description="glm-ocr Stage-1 caption alignment",
        config_name=_cfg_name("align", "stage1caption"),
        timeout_min=720,
    )


def _stage2() -> IterPlan:
    return IterPlan(
        cmd=_train("--dataset", "mix", "--objective", "qa", "--mixture", INSTRUCT_MIX, "--vqa", DOC_VQA,
                   "--unfreeze", "lora", "--lora-rank", "128", "--init-projector", str(STAGE1_CKPT),
                   "--name-suffix", "glmocr_docstage2", "--steps", "3000", "--n-train", "16000",
                   "--qa-eval-n", "160", "--resume",
                   "--description", "GLM-OCR Stage-2 doc reading, warm-started from Stage-1. Outcomes = "
                   "docvqa/ocrvqa/infographic_vqa/visualmrc; vqav2 = glyph-independent grounding control."),
        description="glm-ocr Stage-2 doc reading (warm-started)",
        config_name=_cfg_name("mix", "glmocr_docstage2"),
        timeout_min=300,
    )


class GlmOcrPipeline:
    """Stage-1, then Stage-2 only if Stage-1 grounded (best.pt produced). Lazy generator: the gate is
    evaluated after Stage-1 finishes, so it sees the real on-disk checkpoint."""

    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]:
        yield _stage1()
        # history[:] is refreshed between iters, so it now holds Stage-1's outcome row. Run Stage-2 only
        # if Stage-1 produced a checkpoint AND completed cleanly — a triage-killed/crashed Stage-1 can
        # leave an early best.pt, and we must not warm-start Stage-2 from a hung/undersized run.
        killed = bool(history) and history[-1].get("status") in ("EARLY_KILL", "CRASH")
        if STAGE1_CKPT.exists() and not killed:
            print(f"[gate] Stage-1 grounded cleanly ({STAGE1_CKPT.name}) -> running Stage-2", flush=True)
            _janitor()  # reap any orphans Stage-1 left before Stage-2 binds the GPU
            yield _stage2()
        else:
            why = "killed/crashed" if killed else "no best.pt"
            print(f"[gate] Stage-1 did NOT cleanly ground ({why}) -> skipping Stage-2", flush=True)


class JsonlExtractor:
    """train.py already appends the outcome row (KEEP/CRASH + GPU stats) to results.jsonl in its finally
    hook; hand the latest row back to SweepRunner so it lands in history / the sweep summary."""

    def extract(self, plan: IterPlan, run_id: str | None, exit_code: int) -> list[dict[str, Any]]:
        if plan.config_name:
            rows = load_results(experiments_dir="experiments", tag=TAG, config_name=plan.config_name)
            if rows:
                return rows[-1:]
        return [{"status": "CRASH", "notes": f"exit={exit_code}, no results row"}]


def _janitor() -> None:
    """Reap orphaned PPID=1 processes (e.g. train.py DataLoader workers left after a triage kill) so they
    don't hold GPU memory into the next iter. Best-effort: skip cleanly if the CLI isn't available."""
    try:
        out = subprocess.run([sys.executable, "-m", "autoresearch.janitor", "--apply"],
                             capture_output=True, text=True, timeout=120, cwd=str(REPO))
        tail = (out.stdout or out.stderr).strip().splitlines()[-3:]
        print("[janitor] " + " | ".join(tail) if tail else "[janitor] nothing to reap", flush=True)
    except Exception as e:  # noqa: BLE001 — cleanup is best-effort, never block the sweep
        print(f"[janitor] skipped ({type(e).__name__}: {e})", flush=True)


def _wait_for_gpu() -> None:
    """Queue behind the running cheap arm: wait until it finishes and GPU memory drains, so we don't
    contend on the single A100."""
    print(f"[queue] waiting for the cheap arm + GPU to free {time.strftime('%H:%M:%SZ', time.gmtime())}",
          flush=True)
    while subprocess.run(["pgrep", "-f", "run_glmocr_docscratch"], capture_output=True).returncode == 0:
        time.sleep(120)
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        try:
            if int(out[0]) < 5000:
                break
        except (ValueError, IndexError):
            pass  # transient/odd nvidia-smi output — retry rather than crash the wait loop
        time.sleep(60)
    print("[queue] GPU free — starting pipeline", flush=True)


def main() -> None:
    os.chdir(REPO)
    _wait_for_gpu()
    triage = GPUTriageMonitor(
        thresholds=GPUTriageThresholds(
            grace_s=600,            # glm-ocr model load + 200k-sample preload + first eval before triage arms
            hang_util_pct=8, hang_window_s=300,        # <8% for 5min  -> hang
            wasted_util_pct=25, wasted_window_s=1200,  # <25% for 20min -> wasted (relaxed for heavy encode)
            undersized_mem_pct=50, undersized_window_s=1800,  # <50% for 30min -> undersized
        ),
        poll_interval_s=10,
    )
    runner = SweepRunner(tag=TAG, planner=GlmOcrPipeline(), extractor=JsonlExtractor(), triage=triage,
                         experiments_dir="experiments", iter_timeout_min=720, triage_poll_s=10,
                         pause_between_iters_s=30)
    result = runner.run()
    _janitor()
    print(f"\n=== glm-ocr pipeline: {result.iterations} iters, {result.kills} killed ===", flush=True)
    for o in result.outcomes:
        print(f"  {o.plan.description}: {o.kill_reason or 'OK'} (exit={o.exit_code}, {o.elapsed_s:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
