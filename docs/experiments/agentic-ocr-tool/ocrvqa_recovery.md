# ocrvqa-recovery — undo the tool-loop GSPO collapse without losing the +33%

> `feat/ocrvqa-tool-recovery` · PR [#40](https://github.com/charleneleong-ai/lagunaxs2-rlvr/pull/40) · follows the tool-loop GSPO win in [#39](https://github.com/charleneleong-ai/lagunaxs2-rlvr/pull/39).
> Sweep: [`configs/schedules/ocrvqa_recovery.yaml`](../../../configs/schedules/ocrvqa_recovery.yaml) → [`scripts/ocrvqa_recovery_sweep.py`](../../../scripts/ocrvqa_recovery_sweep.py).
> **Pre-registered** — predictions below are written before any run; results/verdict fill in once the sweep lands.

## Diagnosis (the regression this targets)

Tool-loop GSPO ([#39](https://github.com/charleneleong-ai/lagunaxs2-rlvr/pull/39)) lifted overall greedy solve-rate **0.129 → 0.171 (+33% rel)** but regressed **ocrvqa 0.275 → 0.025**. From the 280-episode probe replies + the trainer code:

- GSPO drove the `ocr`-call rate to **0% on every corpus**. The spliced image embedding suffices for chart/doc/infographic *reasoning* — direct-answering even *helped* there — but **ocrvqa/textvqa need verbatim transcription only the `ocr` tool surfaces** (SFT called `ocr` 55%/42% there; GSPO never does).
- On ocrvqa the policy then **collapsed to a memorized constant** (`"Samuel William Bloom"` — 3 distinct replies / 40 items).

Two structural mechanisms, both in [`vision_tool_gspo.py`](../../../src/laguna_rlvr/visual/vision_tool_gspo.py):
1. **Signal-swamping** — on an unsolved transcription item the only pro-`ocr` signal is the `+0.1` tool term in [`episode_reward`](../../../src/laguna_rlvr/visual/vision_tool_gspo.py), and cross-batch advantage normalization (±1.0 solve swings from reasoning corpora dominate `batch_std`) shrinks it to noise.
2. **Abandonment** — the [`DifficultySampler`](../../../src/laguna_rlvr/visual/vision_tool_gspo.py) `p(1−p)` weight collapses once ocrvqa's solve-EMA → 0, so it stops being sampled and the policy drifts further.

## Hypothesis

The collapse is *recoverable without sacrificing the good drift* because the good drift (dropping `ocr` on reasoning corpora) and the bad drift (dropping `ocr` on transcription) are separable — the targeted knobs restore the `ocr` gradient **only where the read needs it**, whereas a KL leash dampens both. Each variant warm-starts from the same SFT adapter and shares the baseline config (lr 1e-5, G=8, batch=2, difficulty sampling, 800 steps); only the named knob differs.

## Pre-registered predictions

| variant | knob | mechanism | prediction |
|---|---|---|---|
| `baseline` | — | — | reproduces the regression: overall ≈ 0.17, ocrvqa ≈ 0.02–0.05 (un-confounded control) |
| `sampler-floor` | `--sampler-floor 0.30` | abandonment (2) | partial ocrvqa recovery; keeps it sampled but doesn't fix the swamped gradient → modest lift |
| `corpus-norm` | `--corpus-norm` | swamping (1) | **strongest targeted fix** — an unsolved `ocr`-call goes positive within its corpus; expect the largest ocrvqa lift with overall held |
| `tool-bump` | `--tool-bonus 0.4` | swamping (1) | helps, but the bonus is still cross-batch-centred so partially re-swamped → between floor and corpus-norm |
| `kl-anchor` | `--kl-coef 0.02` | both (blunt) | prevents the constant-answer collapse, but the uniform leash also claws back some reasoning-corpus drift → ocrvqa up, **overall risk** down |

**Verdict rule:** a variant passes iff **overall ≥ 0.171 AND ocrvqa ≥ 0.20**. Headline question: does a *targeted* knob (corpus-norm) beat the *blunt* leash (kl-anchor) on the overall–ocrvqa frontier?

## Results

_Pending — sweep gated on GPU (single A100 held by an unrelated `wm` run). Auto-launches via the GPU watcher when it frees; `scripts/ocrvqa_recovery_sweep.py` prints the ranked table._

| variant | overall | ocrvqa | verdict |
|---|---|---|---|
| baseline | — | — | — |
| sampler-floor | — | — | — |
| corpus-norm | — | — | — |
| tool-bump | — | — | — |
| kl-anchor | — | — | — |

## Verdict

_TBD._

## Next move

_TBD — if a knob clears the bar, fold it into the default GSPO config and re-probe the full 12-task matrix; if only kl-anchor recovers ocrvqa but drops overall, the frontier favours the targeted knob and the leash is dropped._
