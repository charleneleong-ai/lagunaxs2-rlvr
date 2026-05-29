#!/usr/bin/env bash
# One turn of the self-evolving RLVR flywheel, end to end:
#   1. synthesize a VERIFIABLE general-agent corpus (the gate drops tasks whose gold fails its verifier)
#   2. measure the solver's base rate on that corpus (general_agent env, via our probe)
#   3. rank the learnable signal (success-spread) -> which domain is worth RL
#   4. (gate) if learnable, free hosted Laguna RL
#
# Defaults are $0/local (Ollama). For real runs:
#   SYNTH_MODEL=poolside/laguna-xs.2 SYNTH_BASE_URL=https://api.pinference.ai/api/v1 SYNTH_API_KEY=$PRIME_KEY \
#   SOLVER=laguna  bash scripts/rlvr_loop.sh library warehouse clinic
set -euo pipefail

CORPUS="docs/adaption/general_agent_corpus.jsonl"
SYNTH_MODEL="${SYNTH_MODEL:-qwen3:8b}"   # generator
SOLVER="${SOLVER:-ollama}"               # conf/model name for the eval solver (ollama | laguna)
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"

echo "== 1/4 synthesize a verifiable corpus with ${SYNTH_MODEL} (invalid generations are gated out) =="
SYNTH_MODEL="$SYNTH_MODEL" uv run python -m laguna_rlvr.synth "$@"

echo "== 2/4 measure ${SOLVER} base rate on the synthesized corpus =="
uv run python -m laguna_rlvr.probe env=general_agent model="$SOLVER" "env.args.source=$CORPUS"

echo "== 3/4 rank learnable signal -> results/probe/ranking.{md,png} =="
uv run python -m laguna_rlvr.report

echo "== 4/4 gate + train =="
echo "If a domain is learnable (0<base_rate<1), point configs/rl/laguna-xs2.toml [[env]] at general_agent"
echo "(args.source=$CORPUS) and run 'mise run train' for the free hosted Laguna RL run."
