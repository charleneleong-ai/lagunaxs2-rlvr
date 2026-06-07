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

## Per-task accuracy by backbone (all-tasks Stage-2 run each)

All three columns are the **all-tasks** Stage-2 runs (`stage2naflex_alltasks` / `stage2anyres_alltasks` /
`stage2qwen3vl_alltasks`) on the same 12-task VQA eval — so the comparison is apples-to-apples across the full
task set, no dashes. Each cell is **final / peak** = final-step accuracy / best single eval over the run, so a
dead `0.00 / 0.00` means the task never scored *even at its best eval*, while `0.00 / 0.33` is a flicker that
didn't hold. The clean confirmed-recipe NaFlex rerun (`stage2instruct`, qa_best **0.225**, in the verdict table
above) used a narrower 6-task reading suite; it agrees with the column here on the wall (docvqa/ocrvqa = 0) and
scores vqav2 0.60 / visual7w 0.55 on its suite.

Eval subsets are tiny (`qa_eval_n=160` split ~12 ways → ~13 items/task), so single-item flips read as 0.06–0.08
and peak spikes are noise, not stable capability.

NaFlex and AnyRes share the **same SigLIP2 so400m backbone** — NaFlex
([`siglip2-so400m-patch16-naflex`](https://huggingface.co/google/siglip2-so400m-patch16-naflex)) = native
variable-resolution (no tiling), AnyRes
([`siglip2-so400m-patch16-384`](https://huggingface.co/google/siglip2-so400m-patch16-384)) = 384 fixed-square +
AnyRes tiling. Qwen3-VL ([`Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)) = a separate
tower. So NaFlex-vs-AnyRes is a same-encoder resolution A/B.

| Task type | Task | SigLIP2-NaFlex | SigLIP2-AnyRes | Qwen3-VL-4B |
|---|---|---|---|---|
| **Natural-image VQA** | vqav2 | 0.48 / 0.52 | 0.40 / 0.56 | 0.48 / 0.60 |
| (sparse text) | visual7w | 0.57 / 0.71 | **0.64** / 0.64 | **0.64** / 0.64 |
| **Chart/figure VQA** | figureqa | **0.95** / 0.95 | **0.95** / 0.95 | **0.95** / 1.00 |
| (synthetic, sparse) | plotqa | 0.50 / 0.62 | **0.56** / 0.56 | 0.38 / 0.38 |
| | dvqa | 0.06 / 0.19 | 0.06 / 0.19 | 0.12 / 0.12 |
| **Scene-text VQA** | textvqa | 0.23 / 0.23 | 0.23 / 0.31 | 0.23 / 0.31 |
| **Real chart** (dense) | chartqa | **0.33 / 0.33** | 0.00 / 0.00 | 0.00 / 0.33 |
| | chart2text | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| **OCR-dense document** | docvqa | 0.00 / 0.00 | 0.00 / 0.25 | 0.00 / 0.00 |
| | ocrvqa | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| | infographic_vqa | **0.33 / 0.33** | 0.00 / 0.33 | 0.00 / 0.33 |
| | visualmrc | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| **Design code-gen** | websight | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| | webcode2m | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| **Synthetic OCR** | synthetic | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| **Isolation control** | infographic_vqa *(isolated)* | — | **0.32** | — |

## Reading

1. **Grounding is real for sparse-text tasks.** A clean capability gradient by text-density: figureqa 0.95,
   plotqa 0.56, visual7w/vqav2 0.4–0.64 ground well; the intermediate (dvqa) is weak (0.06–0.13); everything
   text-dense is a flat 0.00. The projector can locate a region and read *a* glyph, not *many*.
2. **The truly-dense wall is not resolution.** docvqa / ocrvqa / visualmrc / chart2text are **0.00 final AND
   peak on all three backbones** — NaFlex (native variable res), AnyRes (tiled 384), Qwen3-VL (dynamic high-res)
   never move them. Tower swaps don't help, and (separately) the NaFlex LoRA-rank sweep below doesn't either.
   The only motion on the wall is *peak-flicker* — AnyRes docvqa touches 0.25, Qwen chartqa/infographic touch
   0.33 — but none **hold** (final back to 0). Those spikes are ~1–4 items on a ~13-item subset, i.e. noise.
3. **NaFlex actually wins the semi-dense tasks.** chartqa and infographic_vqa hold **0.33 final** on NaFlex
   where AnyRes/Qwen only flicker there and settle at 0. So NaFlex isn't uniformly behind — the dense gradient is
   chartqa/infographic (semi-dense, partly readable) > docvqa/ocrvqa/visualmrc (the hard floor, dead 0 for all).
4. **Isolation helps the sparse-ish tasks, not dense OCR.** `infographic_vqa` goes **0.00 diluted → 0.32
   isolated** — that capability exists but the 16-task mix crowds it out. But the capacity sweep below shows
   docvqa stays at the noise floor *even isolated and at 4× LoRA rank* — so the wall on truly dense text is a
   reading limit, not a capacity-allocation one.
5. **Backbone barely matters at the wall.** AnyRes-siglip and Qwen3-VL tie at 0.3125 with near-identical
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
3. ~~**Decoder-unfreeze + resolution grid**~~ — **DONE** (2026-06-06). 3×2 grid (decoder plasticity × vision
   tokens), isolated doc family, NaFlex (W&B `*docunf_{moe,topk,attn}_p{1,4}`, all 5 new arms finished). **Result
   below: clean negative — docvqa stays 0.00 final in every cell.** Neither decoder-FFN plasticity nor finer
   resolution breaks the wall. Remaining levers are 8-bit-AdamW routed-expert unfreeze or targeted dense-OCR data.

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

## Decoder-unfreeze + resolution grid — the wall holds (clean negative)

Spec: [`docs/specs/2026-06-06-decoder-unfreeze-ocr-wall-design.md`](../../specs/2026-06-06-decoder-unfreeze-ocr-wall-design.md).
3×2 grid isolating the two remaining levers — **decoder plasticity** (attn-LoRA / lora-moe shared-expert FFN /
top-k top-4-layer full unfreeze, routed experts kept frozen) × **vision tokens** (`--pool 4` / `--pool 1` finer).
Document VQA family isolated (`docvqa,infographic_vqa,visualmrc`), NaFlex, warm-started from the 2026-06-05 Stage-1
caption checkpoint, 3000 steps each. `infographic_vqa` (~0.18 when isolated) is the in-sweep positive control.
A0 = attn-LoRA·pool4 is the reused `dociso_r128` run. Each cell is **final / peak** over the run:

| decoder × vision | docvqa | infographic_vqa *(control)* | visualmrc |
|---|---|---|---|
| attn-LoRA · pool4 *(=A0, `dociso_r128`)* | 0.00 / 0.067 | 0.12 / 0.12 | 0.00 / 0.00 |
| **lora-moe** · pool4 | 0.00 / 0.00 | 0.18 / 0.18 | 0.00 / 0.00 |
| **top-k** · pool4 | 0.00 / 0.00 | 0.06 / 0.06 | 0.00 / 0.00 |
| attn-LoRA · **pool1** | 0.00 / 0.033 | 0.06 / 0.06 | 0.05 / 0.05 |
| **lora-moe** · **pool1** | 0.00 / 0.00 | 0.12 / 0.18 | 0.05 / 0.05 |
| **top-k** · **pool1** | 0.00 / 0.00 | 0.12 / 0.12 | 0.00 / 0.00 |

**Reading — decoder plasticity and finer tokens both fail; it's a base-decoder reading limit.** docvqa is **0.00
final in every cell**, peak only flickering to 0.033–0.067 (1–2 items / ~30) — the same noise floor as the capacity
sweep. Neither lever moves it: (1) decoder-FFN plasticity (lora-moe) and full top-k unfreeze add nothing on
docvqa, and top-k actually *traded down* the one live task (infographic 0.06 vs A0's 0.12) — the extra unfrozen
params compete against the capability that existed. (2) Finer vision tokens (pool 1) don't lift docvqa either;
visualmrc's 0.045 is a single item. The infographic control stayed alive (0.06–0.18) the whole time, so the eval
is sound and the 0.00s are real transcription failures, not a broken harness.

**No winning config to promote** → the all-tasks generalization rerun is moot for this sweep (nothing lifted
docvqa to generalize). The OCR-dense wall has now survived backbone swaps, LoRA-rank 64→256, decoder-FFN
plasticity, top-k full unfreeze, and finer resolution — strong evidence it's a **base-decoder dense-OCR
transcription limit**, not an adapter / capacity / resolution one. Remaining levers are heavier: 8-bit-AdamW
routed-expert unfreeze, or targeted dense-OCR transcription pretraining data.
