#!/usr/bin/env bash
# Submit a free hosted Laguna XS.2 RL run on a verifiers env.
#
#   bash scripts/submit_training.sh [ENV_DIR] [CONFIG]
#   # default:  environments/code_smoke  configs/rl/laguna-xs2.toml       (MBPP code-repair)
#   # read RL:  bash scripts/submit_training.sh environments/ocr_tool configs/rl/laguna-read-gspo.toml
#
# Vendors the two pure laguna_rlvr helpers into the env dir so the hosted container is
# self-contained (no extra install), pushes the env, then launches `prime train`. The vendored copy
# is gitignored and removed on exit — committed source stays DRY.
#
# Uses the team's single free concurrent Laguna run slot. `prime train` will prompt to confirm.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_DIR="${1:-environments/code_smoke}"
CONFIG="${2:-configs/rl/laguna-xs2.toml}"
VENDOR="$ENV_DIR/laguna_rlvr"

cleanup() { rm -rf "$VENDOR"; }
trap cleanup EXIT

echo ">> Vendoring laguna_rlvr/{code_exec,rewards}.py into $ENV_DIR (self-contained for hosted training)"
mkdir -p "$VENDOR"
printf '"""Vendored at push time — minimal helpers for the hosted env."""\n' > "$VENDOR/__init__.py"
cp src/laguna_rlvr/code_exec.py src/laguna_rlvr/rewards.py "$VENDOR/"

echo ">> Sanity-checking the env imports from the vendored copy"
( cd "$ENV_DIR" && python -c "
import sys; sys.path.insert(0, '.')
import laguna_rlvr.code_exec, laguna_rlvr.rewards  # noqa: F401
print('   vendored laguna_rlvr imports OK')
" )

echo ">> Pushing env to the Hub from $ENV_DIR"
prime env push --path "$ENV_DIR"

echo ">> Launching free hosted Laguna RL run from $CONFIG"
prime train "$CONFIG"

echo ">> Submitted. Track with: prime train list   (or the dashboard)."
echo "   Measure lift: prime eval run <env-id> -m poolside/laguna-xs.2  (before vs after; see $CONFIG)"
