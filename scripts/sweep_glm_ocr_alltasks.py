"""GLM-OCR all-tasks Stage-2 — fills the GLM-OCR row of the cross-backbone matrix, like-for-like with the
siglip/qwen `stage2*_alltasks` runs (--tasks vqa,chart,document,design,ocr = 12 VQA + design-codegen + ocr
mix). Warm-started from the GLM-OCR Stage-1 caption ckpt (already grounded), run under autoresearch
SweepRunner with active GPU triage. Resolves the vqav2 confound (clean all-tasks comparison) and shows
whether the broader mix moves any dense task before we consider decoder unfreeze.

    uv run python scripts/sweep_glm_ocr_alltasks.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

from autoresearch.gpu_monitor import GPUTriageThresholds
from autoresearch.sweep_runner import GPUTriageMonitor, IterPlan, SweepRunner

sys.path.insert(0, str(Path(__file__).parent))
from sweep_glm_ocr import REPO, STAGE1_CKPT, TAG, JsonlExtractor, _cfg_name, _janitor, _train  # noqa: E402


class AllTasksPlan:
    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]:
        yield IterPlan(
            cmd=_train("--dataset", "mix", "--objective", "qa",
                       "--tasks", "vqa,chart,document,design,ocr",
                       "--unfreeze", "lora", "--lora-rank", "128",
                       "--init-projector", str(STAGE1_CKPT),
                       "--name-suffix", "glmocr_alltasks_mb1", "--steps", "3000", "--n-train", "16000",
                       "--qa-eval-n", "160", "--resume",
                       "--description", "GLM-OCR all-tasks Stage-2 (12 VQA + design-codegen + ocr mix), "
                       "warm-started from the GLM-OCR Stage-1 caption ckpt. Like-for-like with the "
                       "siglip/qwen stage2*_alltasks matrix arms (only --encoder differs). Fills the "
                       "GLM-OCR matrix row + resolves the vqav2 confound before any decoder unfreeze.",
                       # all-tasks 12-corpus mix OOMs at micro_batch=2/pool=1: micro_batch=1 halves the
                       # decoder forward, pool=2 halves resampler patches. base+optimizer already ~70GB.
                       pool="2", cfg="configs/mm_adapter/a100-80gb-laguna-bf16.toml"),
            description="glm-ocr all-tasks (12 VQA + design + ocr)",
            config_name=_cfg_name("mix", "glmocr_alltasks_mb1"),
            timeout_min=360,
        )


def main() -> None:
    os.chdir(REPO)
    triage = GPUTriageMonitor(
        thresholds=GPUTriageThresholds(
            grace_s=1200,           # heavy 12-corpus preload runs at ~0% util before training — don't false-hang-kill
            hang_util_pct=8, hang_window_s=300,
            wasted_util_pct=25, wasted_window_s=1200,
            undersized_mem_pct=50, undersized_window_s=1800,
        ),
        poll_interval_s=10,
    )
    runner = SweepRunner(tag=TAG, planner=AllTasksPlan(), extractor=JsonlExtractor(), triage=triage,
                         experiments_dir="experiments", iter_timeout_min=360, triage_poll_s=10,
                         pause_between_iters_s=15)
    result = runner.run()
    _janitor()
    print(f"\n=== glm-ocr all-tasks: {result.iterations} iters, {result.kills} killed ===", flush=True)
    for o in result.outcomes:
        print(f"  {o.plan.description}: {o.kill_reason or 'OK'} (exit={o.exit_code}, {o.elapsed_s:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
