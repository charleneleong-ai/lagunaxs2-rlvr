# Bridge-data-recipe — results so far

> Progress update for `feat/bridge-data-recipe`. Source: W&B `chaleong/laguna-mm-adapter`, runs 2026-06-02 → 06-04.
> Plan: [`docs/plans/2026-06-03-bridge-fix-data-recipe.md`](../../plans/2026-06-03-bridge-fix-data-recipe.md) ·
> Spec: [`docs/specs/2026-06-03-bridge-fix-data-recipe-design.md`](../../specs/2026-06-03-bridge-fix-data-recipe-design.md)

## Hypothesis

The vision adapter didn't read because the supervision was guessable — the language prior faked the answer, so
the projector never grounded (loss only ~7% image-dependent). Fix = adopt the reference's data recipe: general
image→caption **Stage-1** (grounds the projector via plain LM loss; captioning is inherently image-dependent)
then diverse image-dependent VQA **Stage-2**. Plain LM loss throughout, no architecture change.

## Verdict: grounding formed — the ~0.044 qa ceiling broke to ~0.31

| Stage-2 run | Encoder | qa_best | vs prior 0.044 ceiling | state |
|---|---|---|---|---|
| `stage2anyres_alltasks` | siglip AnyRes | **0.3125** | **7×** | finished |
| `stage2qwen3vl_alltasks` | Qwen3-VL | **0.3125** | **7×** | finished |
| `stage2instruct` | siglip **NaFlex** | 0.225 | 5× | **crashed** (partial — clean rerun pending) |
| `stage2b_doc_iso` (control) | siglip AnyRes | 0.0437 | — | finished |

Stage-1 caption runs (`align`, `recon`, 200k) report `qa_best = -1` by design (no qa eval at the alignment
stage). The lift appears the moment Stage-2 instruction runs on the caption-aligned checkpoint — confirming the
bridge spec's behavioural success criterion.

## Per-task accuracy by backbone (best Stage-2 run each)

`—` = not in that run's eval set (the NaFlex `stage2instruct` crashed early on the smaller plan VQA set).

| Task type | Task | NaFlex | AnyRes | Qwen3-VL |
|---|---|---|---|---|
| **Natural-image VQA** | vqav2 | 0.43 | 0.40 | **0.48** |
| (sparse text) | visual7w | 0.48 | **0.64** | **0.64** |
| **Chart/figure VQA** | figureqa | — | **0.95** | **0.95** |
| (synthetic, sparse) | plotqa | — | **0.56** | 0.38 |
| | dvqa | — | 0.06 | 0.13 |
| **Scene-text VQA** | textvqa | 0.06 | **0.23** | **0.23** |
| **Real chart** (dense) | chartqa | 0.00 | 0.00 | 0.00 |
| | chart2text | — | 0.00 | 0.00 |
| **OCR-dense document** | docvqa | 0.00 | 0.00 | 0.00 |
| | ocrvqa | 0.00 | 0.00 | 0.00 |
| | infographic_vqa | — | 0.00 | 0.00 |
| | visualmrc | — | 0.00 | 0.00 |
| **Design code-gen** | websight | 0.00 | 0.00 | 0.00 |
| | webcode2m | 0.00 | 0.00 | 0.00 |
| **Synthetic OCR** | synthetic | 0.00 | 0.00 | 0.00 |
| **Isolation control** | infographic_vqa *(isolated)* | — | **0.32** | — |

## Reading

1. **Grounding is real for sparse-text tasks.** A clean capability gradient by text-density: figureqa 0.95,
   plotqa 0.56, visual7w/vqav2 0.4–0.64 ground well; the intermediate (dvqa) is weak (0.06–0.13); everything
   text-dense is a flat 0.00. The projector can locate a region and read *a* glyph, not *many*.
2. **The OCR-dense wall is not resolution.** NaFlex (native variable res), AnyRes (tiled 384), and Qwen3-VL
   (dynamic high-res) are identical 0.00 on docvqa/chartqa/ocrvqa/design. Tower swaps don't move it.
3. **It looks like capacity allocation.** `infographic_vqa` goes **0.00 diluted → 0.32 isolated** — the
   capability exists but the 16-task mix crowds it out. Points the next lever at projector/LoRA capacity +
   curriculum, not more data or pixels.
4. **Backbone barely matters at the wall.** AnyRes-siglip and Qwen3-VL tie at 0.3125 with near-identical
   per-task profiles.

## Next move

1. **Clean rerun of `stage2instruct` (NaFlex)** — it's the one config that crashed mid-run (partial 0.225); the
   AnyRes/Qwen siblings already prove the recipe lands ~0.31. Note: the 200k Stage-1 caption checkpoint was lost
   (not saved as a W&B artifact, local `results/` regenerated), so the clean rerun is the full chained
   Stage-1→Stage-2 pipeline.
2. **Isolation + capacity sweep** to chase the OCR-dense wall: isolate one dense task (docvqa or chartqa) like
   the infographic control, vary `--lora-rank` / vision-token count, and test whether it lifts off 0 the way
   isolation lifted infographic (0→0.32) — separating "capacity-starved" from "genuinely can't read dense text".
