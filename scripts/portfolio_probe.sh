#!/bin/bash
# Portfolio learnability sweep: probe each candidate harness against laguna_m1 ($0), then rank by
# learnable signal (base_rate·(1−base_rate)). Fast/short-horizon envs first so quick results land
# early; agentic_repair (max_turns=8 + code-exec) last. Per-env timeout bounds any hang.
cd /home/ubuntu/lagunaxs2-rlvr || exit 1
export PYTHONUNBUFFERED=1

probe () {  # env  n_examples  timeout_s
  echo "=== PROBE $1 (n=$2) start $(date -u +%H:%M:%SZ) ==="
  if timeout "$3" uv run python -m laguna_rlvr.probe env="$1" model=laguna_m1 \
        probe.num_examples="$2" probe.rollouts_per_example=3; then
    echo "=== $1 DONE $(date -u +%H:%M:%SZ) ==="
  else
    echo "=== $1 FAILED/TIMEOUT rc=$? $(date -u +%H:%M:%SZ) ==="
  fi
}

probe general_agent   2  900     # 2 builtin tasks — tiny, weak signal (noted)
probe frontend_design 7  1500    # 7 builtin designs, short horizon
probe agentic_repair  8  2800    # MBPP bug-repair, max_turns=8 — slowest, last
# code_smoke dropped: redundant with agentic_repair (both MBPP code); repair is the stronger representative.

echo "=== ALL PROBES DONE — ranking $(date -u +%H:%M:%SZ) ==="
uv run python -m laguna_rlvr.report
echo "=== RANKING WRITTEN: results/probe/ranking.md ==="
