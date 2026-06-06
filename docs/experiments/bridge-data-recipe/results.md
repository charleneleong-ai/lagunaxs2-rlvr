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
| `stage2instruct` | siglip **NaFlex** | 0.225 | 5× | **clean rerun confirmed** (full chained, 2026-06-05) |
| `stage2b_doc_iso` (control) | siglip AnyRes | 0.0437 | — | finished |

Stage-1 caption runs (`align`, `recon`, 200k) report `qa_best = -1` by design (no qa eval at the alignment
stage). The lift appears the moment Stage-2 instruction runs on the caption-aligned checkpoint — confirming the
bridge spec's behavioural success criterion.

## Per-task accuracy by backbone (best Stage-2 run each)

`—` = not in that run's eval set (the NaFlex `stage2instruct` eval set is the 6-task reading suite; figureqa/plotqa/dvqa/chart2text/infographic/visualmrc are AnyRes-plan-only). NaFlex column is the clean 2026-06-05 rerun.

| Task type | Task | NaFlex | AnyRes | Qwen3-VL |
|---|---|---|---|---|
| **Natural-image VQA** | vqav2 | **0.60** | 0.40 | 0.48 |
| (sparse text) | visual7w | 0.55 | **0.64** | **0.64** |
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
3. **Isolation helps the sparse-ish tasks, not dense OCR.** `infographic_vqa` goes **0.00 diluted → 0.32
   isolated** — that capability exists but the 16-task mix crowds it out. But the capacity sweep below shows
   docvqa stays at the noise floor *even isolated and at 4× LoRA rank* — so the wall on truly dense text is a
   reading limit, not a capacity-allocation one.
4. **Backbone barely matters at the wall.** AnyRes-siglip and Qwen3-VL tie at 0.3125 with near-identical
   per-task profiles.

## Next move

1. ~~**Clean rerun of `stage2instruct` (NaFlex)**~~ — **DONE** (2026-06-05). Full chained Stage-1→Stage-2
   pipeline reproduced the crashed partial exactly: qa_best **0.225** (vqav2 0.60, visual7w 0.55, OCR-dense flat
   0.00). Confirms the recipe lands NaFlex below the AnyRes/Qwen 0.31 tie, and the OCR-dense wall holds on all
   three towers — clearing the way for the capacity sweep.
2. ~~**Isolation + capacity sweep**~~ — **DONE** (2026-06-05). Isolated the document family
   (`docvqa,infographic_vqa,visualmrc`) on NaFlex, warm-started from the Stage-1 caption checkpoint, swept
   `--lora-rank` ∈ {64,128,256} (W&B `*dociso_r{64,128,256}`, all finished). **Result below: capacity is not the
   bottleneck — docvqa never leaves the noise floor.** Next lever moves to decoder unfreeze / targeted OCR data.

## Isolation + capacity sweep — docvqa stays at the noise floor

Peaks from W&B over 3000 steps (eval subsets are tiny — ~30 docvqa / ~17 infographic items — so values are
item-counts, e.g. 0.033 ≈ 1/30):

| arm | docvqa peak | infographic peak | visualmrc |
|---|---|---|---|
| `dociso_r64` | 0.033 (1/30) @900 | 0.176 (3/17) @1200 | 0.00 |
| `dociso_r128` | 0.067 (2/30) @1200 | 0.118 (2/17) @600 | 0.00 |
| `dociso_r256` | 0.033 (1/30) @300 | 0.176 (3/17) @600 | 0.00 |

**Reading — "genuinely can't read dense text", not "capacity-starved".** docvqa does **not** rise with rank: it
flickers 1–2 correct items across all three arms (non-monotonic — r128 highest, r256 falls back), i.e. the noise
floor. infographic holds ~0.12–0.18 with no rank dependence and stays well under the AnyRes isolation control's
0.32. visualmrc is dead flat at 0.00. Quadrupling the LoRA (64→256) buys nothing on the dense-OCR task — the
adapter isn't capacity-limited, it can't transcribe dense glyphs at all. The next lever is **decoder unfreeze or
targeted OCR-transcription data**, not bigger adapters or more vision tokens.
