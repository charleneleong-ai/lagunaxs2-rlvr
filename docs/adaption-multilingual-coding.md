# Adaption → verifiable multilingual coding RL data for Laguna

**Goal:** use Adaption to grow a small verified seed into a large **non-English Python coding** dataset,
targeting Laguna's *own reported* headroom — **57.7% SWE-bench Multilingual vs 69.9% Verified**. The
dataset feeds `environments/code_smoke` (the multi-turn code-repair env) for free hosted Laguna RL.

Why this shape: the prompt is non-English (multilingual *instruction-following*) but the solution +
tests are **Python**, so it stays in our existing subprocess executor — no per-language toolchains —
while still attacking a real, measurable weakness. The reward is **execution-based** (tests pass), so
every example must be self-verifying.

## Schema (one row per task)

| field | type | used by |
|---|---|---|
| `language` | str (es/fr/de/zh/pt/ja/…) | metadata / stratify |
| `difficulty` | easy \| medium \| hard | metadata / curate |
| `prompt` | str — the task **in `language`** (names the function) | env (`prompt_field`) |
| `reference_solution` | str — Python that solves it | verification only (not shown to the model) |
| `tests` | list[str] — `assert`-style checks | env (`tests_field`) |
| `setup` | str — imports/scaffolding prepended before tests (often "") | env (`setup_field`) |

Seed (verified, 6 rows, all langs/difficulties): `docs/adaption/seed_multilingual_coding.jsonl`
→ pushed to **`chaleong/laguna-multilingual-coding-seed`** (private).

## Quality spec (paste into Adaption's "what good looks like")

> Each example is a **self-contained Python coding task whose prompt is written fluently in a
> non-English language** (`language`), and which names the required function. "Good" means:
> 1. **Verifiable** — `reference_solution` passes **all** `tests` when run as `setup + solution + test`.
> 2. **Discriminating** — an empty/naive stub fails at least one test (no trivially-true tests).
> 3. **Consistent** — the function name in `prompt`, `reference_solution`, and `tests` matches.
> 4. **Fluent + faithful** — the prompt reads naturally in `language` and fully specifies the task
>    (no English leakage in the spec; comments/identifiers may stay ASCII).
> 5. **Intermediate difficulty** — solvable by a strong coder but not one-shot trivial; spread across
>    `easy/medium/hard` and across languages (es, fr, de, zh, pt, ja, … — aim for balance).
> 6. ≥3 tests per task, covering a normal case, an edge case, and a boundary.

Reject anything where the reference fails its tests, an empty stub passes, or the prompt is English.

## End-to-end flow

```
docs/adaption/seed_multilingual_coding.jsonl
  └─(pushed)→ HF chaleong/laguna-multilingual-coding-seed   ← Adaption seed input (or upload the file)
       └─ Adaption: apply the spec above → adapt to N rows across languages → evaluate → export
            └─ HF chaleong/laguna-multilingual-coding         ← the adapted dataset
                 └─ env: load_environment(source="hf:chaleong/laguna-multilingual-coding:train")
                      └─ prime env push  →  prime train configs/rl/laguna-xs2.toml   (free hosted RL)
```

Before training, always eval first: `prime eval run chaleong/code-smoke -m poolside/laguna-xs.2`
(set `source` to the adapted dataset via `--env-args`) to confirm Laguna's base rate is *intermediate*
(learnable signal), then `prime train`.
