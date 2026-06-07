# Resampler-bottleneck (representation) sweep vs the OCR-dense wall

> 2026-06-07 · `feat/bridge-data-recipe` · chases the same docvqa=0 wall as the
> [decoder-unfreeze spec](2026-06-06-decoder-unfreeze-ocr-wall-design.md), from the representation side.

## Motivation — the wall is a *representation* squeeze nobody varied

docvqa / ocrvqa / visualmrc / chart2text are **0.00 final on every configuration tried**: 3 encoders
(NaFlex / AnyRes / Qwen3-VL), LoRA rank 64→256, decoder-FFN plasticity (lora-moe), top-k full unfreeze,
and finer vision tokens (pool 1). Two facts found while planning this sweep explain *why* and point at
the one untested lever:

1. **NaFlex hard-caps at 256 patches.** The `siglip2-so400m-patch16-naflex` processor has
   `max_num_patches=256`. A full A4 page at 300 dpi (2480×3508) collapses to ~247 patches at **~190 px
   each** — dense glyphs (~10–30 px) are destroyed *at the encoder*, before any projector.
2. **All three matrix runs used `projector=resampler`**, which emits a *fixed* **256 output latents**
   regardless of input. So every tower — including AnyRes's 2880 patches @768 px and Qwen3-VL's
   OCR-native features — was squeezed to **256 vision tokens** before the frozen decoder. 256 tokens for
   a thousand-glyph page.

So the wall is a serial representation bottleneck (`encoder patch budget → 256-latent resampler →
decoder`) that **no encoder swap escaped**, because the swaps never touched the 256-latent output or the
patch budget. That is what this sweep varies.

## What each tower actually fed the decoder at the wall

| tower | input patches | effective res | → decoder tokens |
|---|---|---|---|
| NaFlex | 256 (hard cap) | ~208 px wide | 256 (resampler) |
| AnyRes (grid 2) | 2880 @ 768² | 768 px | 256 (resampler) |
| Qwen3-VL | dynamic | higher | 256 (resampler) |

## Design — AnyRes grid × resampler n_queries (3 arms)

Encoder = **AnyRes `siglip`** (every tile is native 384², in-distribution — no position-embedding
extrapolation, unlike pushing NaFlex's cap). Both SigLIP2 so400m towers share `d_enc=1152`, so the
resampler **warm-starts from the NaFlex Stage-1 caption ckpt** — the cross-attn / kv / FFN grounding
machinery transfers via `Projector.load_compatible`, and only the resized query bank reinits. Isolated
document VQA family (`docvqa,infographic_vqa,visualmrc`), `--pool 1` (max patches to the resampler),
3000 steps, attn-LoRA recipe held constant so the only variables are grid + n_queries.

| arm | grid (input res) | n_queries (decoder tokens) | isolates |
|---|---|---|---|
| `base_g2q256` | 2 (768 px) | 256 | reproduces the failed matrix config (in-sweep, isolated) |
| `g2q1024` | 2 (768 px) | **1024** | *the 256-latent squeeze alone* (same input, 4× output) |
| `g4q1024` | **4 (1536 px)** | **1024** | full representation (glyph-readable input + wide latents) |

**Reading.** `base→g2q1024` answers "is the 256-latent squeeze the wall?"; `g2q1024→g4q1024` answers
"does glyph-readable resolution help once the squeeze is gone?". `infographic_vqa` (~0.32 isolated) is
the in-sweep positive control. **Lift = docvqa final ≥ 0.10 over ≥2 consecutive evals** (not a peak
flicker — the AnyRes-touched-0.25-and-dropped lesson).

## Decision

- **Any arm lifts docvqa ≥0.10 sustained** → representation *was* the wall. Disentangle (which of grid /
  n_queries), then the REQUIRED all-tasks generalization rerun of the winner (survives ~1/12 dilution +
  sparse-task regression guard).
- **`g4q1024` still = 0** → representation is **not** the wall. Up to 1536 px glyph-readable input and
  1024 decoder tokens, the frozen Laguna decoder still can't transcribe → the wall is the base decoder.
  Pivot to the agentic / OCR-tool route (give the model an OCR tool in its harness) rather than chasing
  native end-to-end reading.

## Code

`--grid` (AnyRes tiles, `encoders.load_encoder`), `--n-queries` (resampler output,
`projector.Projector`/`Resampler`), and `Projector.load_compatible` (tolerant warm-start across a
query-bank resize) — committed in `feat(visual): expose resampler n_queries + AnyRes grid`.

## Harness

`logs/run_resampler_bottleneck_sweep.sh` — detached daemon (PPID=1), 6 retries/arm, `--resume`,
`qa-eval-n 160`. Same orchestration as the decoder-unfreeze sweep.
