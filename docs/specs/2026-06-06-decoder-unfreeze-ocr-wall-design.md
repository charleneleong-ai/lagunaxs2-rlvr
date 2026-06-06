# Decoder-unfreeze experiment — breaking the OCR-dense wall

> Design spec for `feat/bridge-data-recipe`. Follows the isolation+capacity sweep
> ([`docs/experiments/bridge-data-recipe/results.md`](../experiments/bridge-data-recipe/results.md)).

## Motivation

docvqa / ocrvqa / visualmrc / chart2text are **0.00 final AND peak on all three backbones** (NaFlex / AnyRes /
Qwen3-VL) — never a single correct item even at their best eval. Two levers have already been ruled out:

1. **Resolution** — three encoders (native-variable / tiled-384 / dynamic-high-res) give identical 0.00.
2. **Attention capacity** — the NaFlex LoRA-rank sweep {64,128,256} left docvqa at the noise floor, non-monotonic.

A third is **already pulled**: the Stage-1 align mix is `synthetic 0.6 + cauldron_rendered_text 0.2 + websight
0.2` ([`corpora.py:128`](../../src/laguna_rlvr/visual/corpora.py#L128)) — dense-text transcription supervision is
in training (SyntheticOCR exact-glyph targets + real rendered-text + IAM handwriting). The code comment at
[`corpora.py:122`](../../src/laguna_rlvr/visual/corpora.py#L122) records that scaling OCR data *"barely steers the
decoder"*, and `synthetic` still evals 0.00. So the wall held despite dense-OCR data at scale.

What is **not** ruled out, and what this experiment tests:

- **Decoder plasticity.** The decoder can *attend* to vision tokens (grounding works; sparse VQA 0.4–0.95) but
  the LoRA only touches `q/k/v/o` — the **MoE expert MLPs stay frozen**
  ([`model.py:125`](../../src/laguna_rlvr/visual/model.py#L125)). The generation/transcription computation lives
  in the FFN/expert path, which has never been adapted. A frozen FFN can't emit dense glyph sequences no matter
  how well attention selects them.
- **Token representation.** The Perceiver resampler emits a fixed token count and `--pool 4` mean-pools encoder
  patches 4× before the resampler ([`train.py:162`](../../src/laguna_rlvr/visual/train.py#L162)). A dense page's
  glyphs may not survive into the vision tokens at all.

## Architecture facts (Laguna-XS.2)

40 layers · hidden 2048 · **256-expert MoE, top-8, + a shared expert** (`shared_expert_intermediate_size 512`) ·
GQA 48/8. The MoE shape dictates the decoder lever: LoRA on all 256 sparse experts is impractical (most never
activate under top-8 routing, so adapters get ~no gradient). The **always-on path is the shared expert + router**
— that is where decoder-MLP plasticity is injected cheaply.

## Hypothesis

The frozen expert/FFN path is the wall: the decoder cannot *transcribe* dense text even when it attends to it and
even when trained on OCR targets. Giving the FFN path plasticity (shared-expert/router LoRA, or full unfreeze of
the top layers) breaks docvqa off 0 — unless the glyphs were never in the tokens, in which case raising vision
resolution (`--pool 1`) is what lifts it. The factorial separates the two.

## Design — 3×2 grid, isolated on the document family

Decoder mechanism × vision resolution, same harness as the capacity sweep (NaFlex, document family isolated,
warm-started from the Stage-1 caption `best.pt`, 3000 steps, `--qa-eval-n 160`, `val_every 300`).

| arm | decoder adaptation | `--pool` | tests |
|---|---|---|---|
| `attn_p4` (control) | attn-LoRA `q/k/v/o` (= current) | 4 | reproduces the wall |
| `moe_p4` | attn + **shared-expert + router LoRA** | 4 | decoder FFN plasticity (low-rank, always-on) |
| `topk_p4` | **top-K layers full unfreeze** | 4 | decoder FFN plasticity (full-rank, output layers) |
| `attn_p1` | attn-LoRA only | 1 | glyphs-in-tokens (representation alone) |
| `moe_p1` | attn + shared-expert + router LoRA | 1 | FFN plasticity + representation |
| `topk_p1` | top-K full unfreeze | 1 | both levers, max |

`attn_p4` ≈ the existing `dociso_r128` run; rerun fresh for a same-protocol baseline. LoRA rank fixed at 128
(the sweep's best attn rank) so the decoder dimension varies *what* is adapted, not *how much*.

## Decision matrix (what each outcome means)

- `moe_*` lifts docvqa, `attn_*` flat → **frozen experts were the wall**; shared-expert plasticity is the fix.
- `topk_*` > `moe_*` → low-rank/always-on isn't enough; needs full-rank plasticity in the routed experts.
- `*_p1` lifts regardless of decoder → **representation was the wall**; the fix is resolution/token-count, and the
  decoder lever is a red herring for docvqa.
- only `*+p1` combos lift (interaction) → **need both**: tokens must carry glyphs *and* the FFN must transcribe.
- nothing lifts → wall is deeper (the SigLIP-NaFlex features don't encode dense glyphs). Pivot to the OCR-native
  encoder path (`corpora.py:112` notes a GLM-OCR encoder) or full unfreeze — out of scope here.

**Lift = sustained, not flickered.** Given the peak-vs-final lesson (AnyRes docvqa touched 0.25 and dropped),
require docvqa **final ≥ 0.10 across ≥2 consecutive evals**, not a single-eval spike, to call the wall broken.

## Code prerequisites

1. `--unfreeze lora-moe` (arms `moe_*`): extend
   [`_apply_lora`](../../src/laguna_rlvr/visual/model.py#L123) to add the shared-expert MLP (`gate/up/down`) and
   the router/gate to `target_modules`, after introspecting exact module names via `named_modules()` at build.
   Persist + reload these LoRA tensors through the existing `adapter_state_dict` path (already filters on
   `requires_grad`, so no change needed there).
2. `--unfreeze top-k` + `--unfreeze-layers N` (arms `topk_*`): set `requires_grad_(True)` on `model.layers[-N:]`
   (attn + shared expert + router + norms; **exclude the 256 routed experts** at N>2 for memory, include only if
   N≤2). Extend `adapter_state_dict` to save the unfrozen base tensors.
3. **Differential LR for `topk_*`**: unfrozen pretrained weights need ~10× lower LR than the projector/LoRA
   (projector/LoRA 2e-5, base layers 2e-6) to avoid clobbering base knowledge. Add a param-group split in the
   optimizer ctor; projector + LoRA in the fast group, unfrozen base in the slow group.
4. Vision resolution: `--pool 1` — **no code change** (native flag). Output token count is fixed by the
   resampler's `n_queries`; `--pool` controls how much encoder detail feeds its cross-attention KV. If `_p1` is
   flat, a follow-up `--n-queries` knob (more output tokens) is the deeper representation lever.

## Harness & ops

- `configs/schedules/decoder_unfreeze_dociso.yaml` — 6-arm schedule (`common_overrides` + per-arm `unfreeze` /
  `pool`), driven by the autoresearch orchestrator with the standard triage (step_time / KL / no-learn / GPU
  hang+wasted+undersized).
- Detached daemon, PPID=1 (`setsid nohup … </dev/null >>log 2>&1 & disown`), per the long-running-jobs convention.
- Writeup: `docs/experiments/bridge-data-recipe/decoder_unfreeze_dociso.md` (scaffold with
  `uv run autoresearch-report`).

## Cost & risk

- ~6 arms × ~2.5 h ≈ **15 h sequential** on one A100-80GB; ~7.5 h if split across 2 GPUs.
- **Base drift (`topk_*`)**: differential LR + top-K-only + monitor `embedding_norm_ratio` and guard the sparse
  tasks — vqav2/visual7w must not regress (a collapse there means the base is being clobbered, kill the arm).
- **Memory (`topk_*`)**: routed-expert unfreeze is heavy; start N=4 shared-only, include routed experts only at
  N=2. Sweep triage's undersized/OOM kill covers the rest.
- **Noise floor**: tiny eval subsets — rely on the sustained-lift criterion, not peaks.

## Success criterion

Any arm clears docvqa **final ≥ 0.10 sustained** → the wall is broken and the decision matrix names the lever.
A clean all-flat result is also a result: it eliminates decoder-FFN plasticity + representation and points the
next experiment at the encoder's OCR path.
