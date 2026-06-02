#!/usr/bin/env bash
# Submit a free hosted Laguna XS.2 RL run for one of our envs.
#
#   bash scripts/submit_training.sh [env_name] [config_path]
#   bash scripts/submit_training.sh                                       # default: code_smoke
#   bash scripts/submit_training.sh ocr_tool configs/rl/ocr-tool-mixed.toml
#
# Vendors the pure laguna_rlvr leaf modules into the env dir so the hosted container is
# self-contained (no extra install), pushes the env to the Hub, then launches `prime train`.
# The vendored copy is gitignored and removed on exit — committed source stays DRY.
#
# Uses the team's single free concurrent Laguna run slot. `prime train` will prompt to confirm.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="${1:-code_smoke}"
CONFIG="${2:-configs/rl/laguna-xs2.toml}"
ENV_DIR="environments/$ENV_NAME"
VENDOR="$ENV_DIR/laguna_rlvr"

[ -d "$ENV_DIR" ] || { echo "no env dir at $ENV_DIR" >&2; exit 1; }
[ -f "$CONFIG" ]  || { echo "no config at $CONFIG" >&2; exit 1; }

cleanup() { rm -rf "$VENDOR"; }
trap cleanup EXIT

echo ">> Vendoring laguna_rlvr leaf modules into $ENV_DIR (self-contained for hosted training)"
mkdir -p "$VENDOR"
printf '"""Vendored at push time — minimal pure helpers for the hosted env."""\n' > "$VENDOR/__init__.py"
cp src/laguna_rlvr/code_exec.py src/laguna_rlvr/rewards.py src/laguna_rlvr/scaffold.py "$VENDOR/"
if grep -q "laguna_rlvr.synth" "$ENV_DIR"/*.py 2>/dev/null; then   # general_agent needs the synth task helper
    mkdir -p "$VENDOR/synth"
    : > "$VENDOR/synth/__init__.py"
    cp src/laguna_rlvr/synth/task.py "$VENDOR/synth/"
fi

echo ">> Sanity-checking the env imports from the vendored copy"
( cd "$ENV_DIR" && python -c "
import sys; sys.path.insert(0, '.')
import laguna_rlvr.code_exec, laguna_rlvr.rewards, laguna_rlvr.scaffold  # noqa: F401
print('   vendored laguna_rlvr imports OK')
" )

echo ">> Pushing env to the Hub"
prime env push --path "$ENV_DIR"

echo ">> Launching free hosted Laguna RL run from $CONFIG"
prime train "$CONFIG"

echo ">> Submitted. Track with: prime train list   (or the dashboard)."
echo "   Measure lift: mise run eval -- $ENV_NAME   (before vs after)"
