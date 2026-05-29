# Terminal-Bench × verifiers compatibility (findings 2026-05-29)

Why the local Terminal-Bench env stalled, and the fast paths forward. Captured so a
later session doesn't re-walk the swamp.

## The core problem: verifiers API skew

The `prime` CLI bundles **verifiers 0.1.14**. Its `MultiTurnEnv` contract:
- `rollout(self, input, client, model, sampling_args)` is **`@final`** — cannot be overridden.
- `is_completed` is **`@final`** — completion comes from `@vf.stop`-decorated methods (`max_turns_reached` is built in).
- `env_response(self, messages, state, **kwargs) -> Messages` — returns **Messages only** (state is mutated in place), not a `(messages, state)` tuple.
- `setup_state(self, state) -> None` — provision per-rollout resources here; read per-task data from `state["info"]`.
- Scoring: rubric reward funcs run after the rollout; per-rollout teardown/scoring belongs in a `@vf.cleanup` handler.

The community TB wrappers each target an **older/different** verifiers and break on 0.1.14:

| Wrapper | Version / date | Breakage on 0.1.14 |
|---|---|---|
| `ibrahim/terminal-bench` | 0.4.0 / Sep 2025 | Overrides the `@final` `rollout`; provisions Docker inside it. Hard-incompatible. (This is what our committed `terminal_bench_curated.py` forked — **stale**.) |
| `popfido/terminalbench` | 0.5.2 / Oct 2025 | Closer: provisions in `setup_state`, ships a reusable `DockerExecutor`. But `env_response` returns tuples (needs `-> Messages`) and overrides the `@final` `is_completed`. Needs glue-porting. |
| `primeintellect/terminal-bench-2` | 0.2.1 / May 2026 | **Current** — built on `verifiers.v1` + Harbor, but runs on **hosted `prime-sandboxes` only** (no local Docker; `prime_sandboxes` is an httpx client to Prime's API). Needs event credits. |

Other gotchas hit: `prime env install` is broken (emits `--exclude-newer-package <name>=false`, an invalid uv timestamp) — use `prime env pull` instead; `prime eval run` resolves the env as an *installed package*, so envs must be self-contained; TB dataset version must be a real tag (`terminal-bench-core==0.1.1`, not `head`); `terminal-bench` needs Python ≥3.12 + Docker.

## Fast paths forward (easiest first)

1. **Validate the pipeline with a no-Docker env.** A tiny `verifiers` env over verifiable coding tasks scored in-process (run candidate code against unit tests via subprocess, reward = pass fraction → reuses `rewards.partial_credit`). Runs with local Ollama, $0, no Docker, no terminal-bench. Gets probe→reward→report green *today*. **Recommended first.**
2. **Real TB result at the event: official `terminal-bench-2` + a reward `CallableEntry`.** Don't fork — add `shaped_harbor_reward` (parses `state["harbor_tests"]["stdout"]` → pass fraction) via config, run on sponsored sandboxes. Minimal code.
3. **Local TB (if truly needed): vendor popfido's `DockerExecutor`, write fresh 0.1.14 glue.** ~400 lines, only fully testable via Docker. Highest effort; defer unless local TB is essential.

## Status of `environments/terminal_bench_curated/`

Built on the stale ibrahim API → **does not run on verifiers 0.1.14**. Kept for the reward/curation logic and as a porting reference. Do not `prime eval run` it as-is; pursue path 1 or 2 first.
