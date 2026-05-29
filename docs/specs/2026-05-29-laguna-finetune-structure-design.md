# Laguna XS.2 Finetune — Project Structure Design

**Date:** 2026-05-29
**Context:** Poolside × Prime Intellect hackathon. Fine-tune / post-train [Laguna XS.2](https://huggingface.co/poolside/Laguna-XS.2) (33B MoE / 3B active, agentic coding, Apache-2.0) on Prime Intellect's platform.

## Summary

A two-phase, Hydra-orchestrated workspace for producing **a reusable `verifiers` RL environment + a Laguna RL result** in a single weekend.

- **Phase 1 — Probe (env-first):** measure Laguna's base success rate and reward variance across candidate coding domains; rank by *learnable signal* and pick the target.
- **Phase 2 — RL (stretch):** point one `prime-rl` run at the winning domain + shaped reward.

Hydra is the **orchestration layer only**. It composes configs and shells out to `prime eval run` / `prime-rl`; it never reimplements Prime Intellect logic. The `environments/` stay in PI-native `verifiers` format so they remain portable and Hub-submittable.

### Why this shape (key findings)

- Laguna is **already heavily terminal-RL'd** — it reports **30.1% on Terminal-Bench 2.0** via the open-sourced "pool" ACP harness. Terminal RL is therefore *learnable* (nonzero, unsaturated) but **low-headroom and crowded**.
- Poolside's own numbers expose headroom elsewhere: **57.7% SWE-bench Multilingual vs 69.9% Verified** (~12pt gap) → multilingual / per-language is under-optimized.
- The probe-first approach de-risks the **one-Laguna-run-per-team** limit: never burn the run on a zero-gradient domain.

## Architecture & data flow

```
                 ┌─────────────────── conf/ (Hydra) ───────────────────┐
                 │  model/ · env/ · reward/ · rl/  (composable groups)  │
                 └───────────────┬──────────────────────┬──────────────┘
   PHASE 1: PROBE                ▼                       ▼   PHASE 2: RL
   src/probe.py  ──shell out──►  prime eval run        src/rl.py ──► prime-rl
        │         env × model × reward                      │     (chosen env+reward)
        ▼                                                   ▼
   results/probe/*.jsonl                              results/rl/ (reward curves, ckpt)
        │                                                   │
        ▼                                                   ▼
   src/report.py ──► headroom ranking (base_rate × variance) ──► pick target ──┘
```

- Probe runs each env against a **cheap proxy model** first (plumbing/debug), then **Laguna XS.2**, recording per-task `{success, reward, turns, tool_calls}`.
- Report ranks domains by **`base_rate × variance`** — the metric that predicts whether RL can move the needle (not raw score).
- RL is **gated** behind a passing probe (variance > 0).

## Repo layout

```
laguna-finetune/
├── conf/                          # Hydra config groups (liftr-style)
│   ├── config.yaml                #   defaults + wandb + paths
│   ├── model/                     #   laguna_xs2.yaml · proxy.yaml
│   ├── env/                       #   tb_curated.yaml · multilingual.yaml
│   ├── reward/                    #   binary.yaml · partial.yaml
│   └── rl/                        #   laguna_small.yaml
├── environments/                  # PI-native verifiers envs (own pyproject → Hub-submittable)
│   ├── terminal_bench_curated/
│   └── swe_multilingual/
├── src/laguna_rlvr/
│   ├── probe.py                   # @hydra.main → prime eval run sweep → results/probe/*.jsonl
│   ├── rewards.py                 # pure shaping fns, imported by envs + report
│   ├── report.py                  # aggregate → ranking.{md,png} + wandb
│   └── rl.py                      # @hydra.main → prime-rl on chosen env (zero-gradient guard)
├── results/                       # gitignored (probe outputs, reward curves)
├── docs/
├── tests/
└── pyproject.toml                 # uv; deps: verifiers, prime, hydra-core, omegaconf, wandb
```

## Components

| Module | Responsibility | Interface |
|---|---|---|
| `probe.py` | For each `env × model`, shell out to `prime eval run` (forwarding `--temperature` and the merged `env.args` + `reward` group as `--env-args`, so config is authoritative), parse rollout JSONL, emit per-task records | `python -m laguna_rlvr.probe -m env=tb_curated,multilingual model=proxy,laguna` |
| `rewards.py` | Pure, importable shaping fns. No I/O. | `binary(state)` · `partial_credit(state)` · `efficiency_bonus(state) -> float` |
| `report.py` | Rank `results/probe/*.jsonl` by `base_rate × variance`; wandb summary + PNG | `python -m laguna_rlvr.report` → `results/probe/ranking.{md,png}` |
| `rl.py` | Launch `prime-rl` on chosen env+reward; stream reward curve. Refuses if probe `variance == 0`. | `python -m laguna_rlvr.rl env=<winner> reward=partial rl=laguna_small` |

Environments are standalone `verifiers` packages; they import shaping fns from `rewards.py` but own their dataset + rubric. The entrypoint matches what `prime env init` actually generates (verified 2026-05-29 against CLI 0.6.10 / verifiers 0.1.14): `def load_environment(**kwargs) -> vf.Environment`, returning e.g. a `vf.ToolEnv(dataset=..., rubric=vf.Rubric(funcs=[shaped], weights=[1.0]), max_turns=...)`. The generated env `pyproject.toml` uses a `hatchling` build, declares `tags`, and carries `[tool.verifiers.eval]` defaults (`num_examples`, `rollouts_per_example`) that `probe.py` overrides via flags.

### Config schema

```yaml
# conf/config.yaml
defaults:
  - model: proxy            # default cheap model; opt into laguna explicitly (protects quota)
  - env: tb_curated
  - reward: partial
  - rl: laguna_small
  - _self_
wandb: {project: laguna-finetune, entity: ???}   # set entity before first run
paths: {results: ${hydra:runtime.cwd}/results}
```

Two asserted decisions:
1. `rewards.py` lives in `src/` and envs import it (DRY, testable). If an env is later pushed standalone to the Hub, the shaping fn is vendored in.
2. `proxy` is the **default** model so accidental runs don't consume Laguna quota.

## Testing

| Area | File | Asserted |
|---|---|---|
| Reward shaping | `tests/test_rewards.py` | `partial_credit` monotonic in tests-passed; `efficiency_bonus` bounded [0,1] & decreasing in turns; parametrized over empty/failed/partial states |
| Headroom ranking | `tests/test_report.py` | synthetic JSONL ranks by `base_rate × variance`; zero-variance domain sorts last + flagged |
| RL gradient guard | `tests/test_rl_guard.py` | `rl.py` refuses to launch when chosen env variance == 0 (subprocess mocked) |

`rewards.py` is pure → real unit tests. Orchestration modules test only JSONL parsing + the guard; the genuine env check is `prime eval run <env> -m proxy -n 2` (integration).

## Weekend timebox

- **Fri eve** — scaffold; `uv` + `prime lab setup`; `prime lab doctor` green; wrap `terminal_bench_curated` + `swe_multilingual`; probe with proxy model.
- **Sat AM** — probe with Laguna (small `n`); `report.py` → ranking → pick target.
- **Sat PM** — single Laguna `prime-rl` run on the winner + reward curves.
- **Sun** — writeup; `prime env push` to the Hub; submit.

## Out of scope (separate specs)

- Quantization track — poolside already shipped [`Laguna-XS.2-NVFP4`](https://huggingface.co/poolside/Laguna-XS.2-NVFP4).
- On-policy distillation.
- More than two candidate domains.
- ARC-AGI-3 env — text-only modality risk + ~0.51% frontier ceiling (zero-gradient trap).

## Implementation notes (2026-05-29)

- **Env API has two generations.** Classic `verifiers` (`MultiTurnEnv`/`Rubric`, local Docker) vs `verifiers.v1` + Harbor (official `primeintellect/terminal-bench-2`, TB-2.1, **hosted `prime-sandboxes` only** — `prime_sandboxes` is an httpx client to Prime's API, no local Docker backend). The orchestration layer is API-agnostic (shells out to `prime`), so the choice only affects env modules.
- **`terminal_bench_curated` is built on the classic local-Docker path** (forked from community `ibrahim/terminal-bench`, MIT) so it runs **credit-free locally** pre-event. Two changes vs upstream: store per-test `tests_passed`/`tests_total` in state (real success key is `terminalbench_is_resolved`), and a shaped partial-credit reward + curated `task_ids`. Event-time upgrade path: port the same curation + a `shaped_harbor_reward` `CallableEntry` (parsing `state["harbor_tests"]["stdout"]`) to the v1/Harbor `terminal-bench-2` for a blessed TB-2.1 run on sponsored sandboxes.
- **Envs are self-contained** (vendor the reward math; declare their own deps) because `prime eval run` resolves the env as an *installed package* (`prime env install <path>` first) — they can't rely on importing the parent `src/` package.
- **Wallet is $0 pre-event**; iterate via `--provider anthropic` (your Anthropic bill, tiny) or local Docker. `--provider prime` / hosted sandboxes wait for sponsored event compute. Username unset → set before `prime env push`.

## References

- Setup: [lab-cookbook](https://github.com/PrimeIntellect-ai/lab-cookbook) (guides 00-setup, 02-building-your-first-environment, 03-training-with-rl, 10-coding-agents-and-sandboxes), hackathon repo [poolside-primeintellect](https://github.com/poolside-primeintellect/poolside-primeintellect).
- Stack: [verifiers](https://github.com/PrimeIntellect-ai/verifiers), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).
- RL precedent: [Orca-Agent-RL](https://github.com/Danau5tin/Orca-Agent-RL) (+160% on TerminalBench, Qwen3-14B base).
- Existing TB wrappers (reuse, don't rebuild): [`ibrahim/terminal-bench`](https://app.primeintellect.ai/dashboard/environments/ibrahim/terminal-bench).
