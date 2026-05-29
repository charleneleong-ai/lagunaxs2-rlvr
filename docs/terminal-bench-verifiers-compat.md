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

## Running the official `terminal-bench-2` (path 2) — verified against live infra 2026-05-29

Got it running end-to-end on credit (env loads, hosted sandbox provisions, results upload to the Prime dashboard). Three layers had to be peeled, in order — recorded so the event run is fast:

1. **Dev verifiers, isolated.** It needs `verifiers>=0.1.15.dev11` (which requires **Python ≥3.12**) for `verifiers.v1` + `Terminus2`/`HarborTaskset`. Pinning that in our project conflicts with `requires-python>=3.11` and destabilizes the 0.1.14 pipeline — so run it in a **throwaway venv**, not the project:
   ```bash
   cd /tmp && uv venv --python 3.12 tb2/.venv && cd tb2
   uv pip install --python .venv --prerelease=allow "verifiers>=0.1.15.dev11" prime-sandboxes
   prime env pull primeintellect/terminal-bench-2          # `prime env install` is broken (bad --exclude-newer-package)
   uv pip install --python .venv -e terminal_bench_2
   source .venv/bin/activate
   ```
2. **The model must be CLOUD-reachable.** `Terminus2` runs the agent *inside* the hosted sandbox, so a local Ollama at `localhost:11434` is unreachable → `SandboxError('command exited with 1')`. Use a Prime-inference model.
3. **In-sandbox agent auth.** The agent calls the model via **litellm** using `OPENAI_MODEL`; pointing it at `deepseek/...` made litellm hit DeepSeek's own API → `AuthenticationError`, and it looked for a missing `configs/endpoints.toml`. The agent needs Prime's inference **base_url + API key set inside the sandbox** (or an `endpoints.toml`) so litellm routes through `api.pinference.ai` with auth. **This is the piece to get from Prime's technical onboarding.**

**Laguna XS.2 caveat:** `poolside/Laguna-XS.2` is on `prime train models` ($0) but **404s on `api.pinference.ai` (inference)** pre-event — its serving is event-gated. So the actual Laguna run is an event-time activity. (`prime inference models` lists what's servable now: deepseek/gemini/llama/qwen/etc.)

Integration note: its `load_environment(config: vf.EnvConfig, *, max_turns)` takes a structured config, not our probe's `--env-args` kwargs — run it directly first, then wire `conf/env/terminal_bench_2.yaml` + a probe adapter. Our partial-credit contribution there = a `shaped_harbor_reward` `CallableEntry` parsing `state["harbor_tests"]["stdout"]`.

## Status

`environments/terminal_bench_curated/` was **removed** (commit after 8237355) — it forked the stale ibrahim API and can't run on 0.1.14. The reward/curation logic lives on in `src/laguna_finetune/rewards.py` (+ tests). Pursue path 1 (no-Docker env) or path 2 (official v1 Harbor + reward `CallableEntry`).
