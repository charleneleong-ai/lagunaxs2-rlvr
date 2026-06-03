# Bridge fix: adopt the reference's data recipe

## Problem

The Laguna-XS.2 vision adapter (frozen encoder → trainable projector(+LoRA) → frozen decoder) does not read. Across every prior configuration `synthetic` reading is a true 0.00, and a diagnostic chain pinned the cause:

- The **decoder is fine** — it copies text from its prompt context 6/6 (`logs/diag_base_decoder.py`).
- The **encoder is fine** — a linear probe reads 20 rendered words at 100% off frozen SigLIP features (`logs/diag_encoder_probe.py`). Glyph signal is present and linearly decodable.
- The **bridge is the failure** — the projector receives healthy gradient (norm 239) but the loss is only **~7% image-dependent** (`logs/diag_projector_grad.py`). The model minimized loss via the language prior `P(answer)`, not `P(answer|image)`; the projector never grounded.

The root cause is **data, not loss or architecture**. Our supervision was guessable: templated synthetic OCR ("The quick brown fox" + a number) and un-learnable `<title>` needles (off-page). The language prior could fake it, so the gradient never forced the projector to convey vision. We also skipped the one step that reliably grounds a projector: general image-caption alignment.

The reference adapter (aaronkazah/laguna-vision) grounds with plain next-token LM loss and no auxiliary objective — because its data forces image-dependence: an LLaVA-pretrain captioning Stage-1 then diverse instruction tuning. Captioning cannot be done without looking, so the standard gradient grounds the projector.

## Approach

Adopt the reference's recipe with our existing loaders. Change *what we train on*, not *how*:

1. **Stage-1 caption alignment** (projector-only, `recon` objective): general image→caption, the grounding bootstrap. Inherently image-dependent, so the LM gradient forces the projector to convey vision.
2. **Stage-2 diverse instruction** (projector + LoRA, `qa` objective): general VQA + our reading VQA + the design corpora, replacing reliance on the un-learnable title-needles.

Plain LM loss throughout. No contrastive term, no architecture change. The contrastive grounding loss considered earlier is deferred: the reference is an existence proof that the right data grounds without it; it is a later enhancement only if grounding forms but plateaus low.

## Data

Sourced from `HuggingFaceM4/the_cauldron` via the existing [`_cauldron`](../../src/laguna_rlvr/visual/corpora.py) factory and [`HFImageTextDataset`](../../src/laguna_rlvr/visual/hf_image_text.py) — one source, no new dataset hosting (LLaVA-Pretrain / ShareGPT4V would need separate image hosting; out of scope).

the_cauldron has no `coco`/`gqa` config (verified — it is VQA-heavy). The real general-caption and general-VQA configs are used instead. New `REGISTRY` corpora (via `_cauldron`):
- `cauldron_localized_narratives` — general dense image captions (Stage-1 grounding core)
- `cauldron_screen2words` — screenshot → summary (UI captioning; grounds the design domain)
- `cauldron_vqav2`, `cauldron_okvqa`, `cauldron_visual7w` — general VQA (Stage-2 image-dependent instruction)

(`cauldron_textcaps` and `cauldron_rendered_text` already exist in `REGISTRY`.)

New mixes:
- `_ALIGN_CAPTION_MIX` — caption-dominant: `cauldron_localized_narratives` + `cauldron_textcaps` + `cauldron_screen2words` + a `synthetic` / `cauldron_rendered_text` reading slice.
- `_INSTRUCT_MIX` — `cauldron_vqav2` + `cauldron_okvqa` + `cauldron_visual7w` + the VQA suite (textvqa/docvqa/chartqa/ocrvqa) + `websight` / `webcode2m` (design).

the_cauldron rows are `{images, texts:[{user, assistant}]}`. The implementation extends [`_cauldron_rows`](../../src/laguna_rlvr/visual/hf_image_text.py) (currently transcription-only) to read the turn structure: assistant text as the caption label for the caption configs (`recon`), and (image, user-question, assistant-answer) for the VQA configs (`qa`).

## Training

Two runs, NaFlex encoder (`siglip_naflex`, pool=4 — cheap, comparable to prior runs; resolution is already ruled out):

1. caption-align → writes `best.pt`
2. `--init-projector <align best.pt>` → instruct + LoRA

Stage-1 captioning needs a caption prompt ("Describe this image.") rather than the transcription `_PROMPT`; a small addition to the `recon` prompt path in [`model.py`](../../src/laguna_rlvr/visual/model.py).

## Success metric

Re-run the bridge diagnostics on the **caption-aligned Stage-1 checkpoint**:
- `logs/diag_projector_grad.py` — loss image-dependence should rise well above the current ~7%.
- `logs/diag_base_decoder.py` — huge-font reads should start landing (vs the current 0/6).

Then Stage-2 per task: reading `qa_acc` (on the fixed word-boundary metric) and design `code_valid` / CodeBLEU, vs the current 0.00.

## Testing

- Loader tests: `cauldron_coco` yields (image, caption); `cauldron_vqav2` / `cauldron_gqa` yield (image, question, answer); the new mixes parse via `parse_mixture` / `load_text_image`.
- Smoke: a caption batch flows through `forward()` and a VQA batch through `forward_qa()` without shape errors.

## Out of scope

- LLaVA-Pretrain / LLaVA-instruct / ShareGPT4V / GQA-native (separate image hosting) — the_cauldron covers caption + VQA from one source.
- The contrastive grounding loss (deferred enhancement).
- AnyRes encoder / resolution (ruled out; a separate axis if needed).
