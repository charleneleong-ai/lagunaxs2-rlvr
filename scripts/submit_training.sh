#!/usr/bin/env bash
# Submit a free hosted Laguna XS.2 RL run on one or more verifiers envs.
#
#   bash scripts/submit_training.sh [CONFIG] [ENV_DIR...]
#   # default:    configs/rl/laguna-xs2.toml  environments/code_smoke           (MBPP code-repair)
#   # read RL:    bash scripts/submit_training.sh configs/rl/laguna-read-gspo.toml environments/ocr_tool
#   # curriculum: bash scripts/submit_training.sh configs/rl/laguna-curriculum.toml \
#                   environments/ocr_tool environments/frontend_design
#
# For each env: vendors laguna_rlvr/{code_exec,rewards}.py in (self-contained for hosted training),
# sanity-imports, and `prime env push`. Then launches `prime train CONFIG` once. Vendored copies are
# gitignored and removed on exit. Uses the team's free Laguna run slot; `prime train` prompts to confirm.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/rl/laguna-xs2.toml}"
shift || true
ENV_DIRS=("$@")
[ ${#ENV_DIRS[@]} -eq 0 ] && ENV_DIRS=("environments/code_smoke")

cleanup() { for d in "${ENV_DIRS[@]}"; do rm -rf "$d/laguna_rlvr"; done; }
trap cleanup EXIT

for ENV_DIR in "${ENV_DIRS[@]}"; do
  VENDOR="$ENV_DIR/laguna_rlvr"
  echo ">> [$ENV_DIR] vendoring laguna_rlvr/{code_exec,rewards}.py (self-contained for hosted training)"
  mkdir -p "$VENDOR"
  printf '"""Vendored at push time — minimal helpers for the hosted env."""\n' > "$VENDOR/__init__.py"
  cp src/laguna_rlvr/code_exec.py src/laguna_rlvr/rewards.py "$VENDOR/"

  echo ">> [$ENV_DIR] sanity-checking the env imports from the vendored copy"
  ( cd "$ENV_DIR" && python -c "
import sys; sys.path.insert(0, '.')
import laguna_rlvr.code_exec, laguna_rlvr.rewards  # noqa: F401
print('   vendored laguna_rlvr imports OK')
" )

  echo ">> [$ENV_DIR] pushing env to the Hub"
  prime env push --path "$ENV_DIR"
done

echo ">> Launching free hosted Laguna RL run from $CONFIG"
prime train "$CONFIG"

echo ">> Submitted. Track with: prime train list   (or the dashboard)."
echo "   Measure lift: prime eval run <env-id> -m poolside/laguna-xs.2  (before vs after; see $CONFIG)"
