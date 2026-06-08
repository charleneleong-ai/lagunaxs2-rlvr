"""GLM-OCR all-tasks Stage-2 WITH decoder unfreeze (lora-moe = shared-expert FFN plasticity) — the last
native shot at the OCR wall: GLM-OCR finally supplies glyph-rich features, so this tests whether the
decoder, given good glyph features AND FFN plasticity, can learn to transcribe (the prior unfreeze sweep
failed on NaFlex, whose features may have lacked glyphs). Warm-started from the GLM-OCR Stage-1 ckpt,
micro_batch=1 + pool=2 (the all-tasks mix OOMs otherwise), QUEUED behind the running attn-LoRA all-tasks
run so it doesn't contend on the single A100.

    uv run python scripts/sweep_glm_ocr_alltasks_unfreeze.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from autoresearch.gpu_monitor import GPUTriageThresholds
from autoresearch.sweep_runner import GPUTriageMonitor, IterPlan, SweepRunner

sys.path.insert(0, str(Path(__file__).parent))
from sweep_glm_ocr import REPO, STAGE1_CKPT, TAG, JsonlExtractor, _cfg_name, _janitor, _train  # noqa: E402

MB1_CFG = "configs/mm_adapter/a100-80gb-laguna-bf16.toml"  # micro_batch=1


class AllTasksUnfreezePlan:
    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]:
        yield IterPlan(
            cmd=_train("--dataset", "mix", "--objective", "qa",
                       "--tasks", "vqa,chart,document,design,ocr",
                       "--unfreeze", "lora-moe", "--lora-rank", "128",
                       "--init-projector", str(STAGE1_CKPT),
                       "--name-suffix", "glmocr_alltasks_moe", "--steps", "3000", "--n-train", "16000",
                       "--qa-eval-n", "160", "--resume",
                       "--description", "GLM-OCR all-tasks Stage-2 + decoder unfreeze (lora-moe shared-expert "
                       "FFN). Last native shot: glyph-rich GLM-OCR features + decoder FFN plasticity. "
                       "Warm-started from the GLM-OCR Stage-1 caption ckpt; compare vs the attn-LoRA "
                       "glmocr_alltasks_mb1 row — does FFN plasticity move docvqa/ocrvqa off 0?",
                       pool="2", cfg=MB1_CFG),
            description="glm-ocr all-tasks + lora-moe decoder unfreeze",
            config_name=_cfg_name("mix", "glmocr_alltasks_moe"),
            timeout_min=480,
        )


def _wait_for_gpu() -> None:
    """Queue behind the running attn-LoRA all-tasks run: wait until its orchestrator exits and GPU drains."""
    print(f"[queue] waiting for glmocr_alltasks_mb1 + GPU {time.strftime('%H:%M:%SZ', time.gmtime())}",
          flush=True)
    while subprocess.run(["pgrep", "-f", r"sweep_glm_ocr_alltasks\.py"],
                         capture_output=True).returncode == 0:
        time.sleep(120)
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        try:
            if int(out[0]) < 5000:
                break
        except (ValueError, IndexError):
            pass
        time.sleep(60)
    print("[queue] GPU free — starting decoder-unfreeze all-tasks run", flush=True)


def main() -> None:
    os.chdir(REPO)
    _wait_for_gpu()
    triage = GPUTriageMonitor(
        thresholds=GPUTriageThresholds(
            grace_s=1200, hang_util_pct=8, hang_window_s=300,
            wasted_util_pct=20, wasted_window_s=1200,   # micro_batch=1 + heavy glm encode runs at lowish util
            undersized_mem_pct=50, undersized_window_s=1800,
        ),
        poll_interval_s=10,
    )
    runner = SweepRunner(tag=TAG, planner=AllTasksUnfreezePlan(), extractor=JsonlExtractor(), triage=triage,
                         experiments_dir="experiments", iter_timeout_min=480, triage_poll_s=10,
                         pause_between_iters_s=15)
    result = runner.run()
    _janitor()
    print(f"\n=== glm-ocr all-tasks+unfreeze: {result.iterations} iters, {result.kills} killed ===",
          flush=True)
    for o in result.outcomes:
        print(f"  {o.plan.description}: {o.kill_reason or 'OK'} (exit={o.exit_code}, {o.elapsed_s:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
