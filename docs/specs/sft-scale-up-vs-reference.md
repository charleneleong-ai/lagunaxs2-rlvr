# SFT scale-up plan — close the ~100× data gap to the reference

## Why

The Laguna-XS.2 vision adapter confabulates because the bridge is ~100× under-trained, not because of
RL / encoder / decoding (see session 2026-06-02). An external reference implementation (internal) is the
**same architecture** (SigLIP-so400m + AnyRes + resampler→256 tokens + LoRA, frozen Laguna, 2-stage SFT)
and reads — because it runs a proper Stage-1 alignment + Stage-2 instruction tuning at **200k** scale.
Ours (`qasftsigliplora`) ran **2,048** examples and skipped LLaVA-Pretrain alignment entirely.

## Recipe gap (reference → ours)

| | Reference | Ours (`qasftsigliplora`) |
|---|---|---|
| Stage-1 data | **LLaVA-Pretrain 120k** (BLIP caps / LAION-CC-SBU) | none — went straight to web/OCR QA |
| Stage-1 cfg | projector-only, LR **1e-3**, 1 epoch, warmup 0.03, max-tiles 1 | projector-only, LR 1e-3 (matches) |
| Stage-2 data | LLaVA-Instruct 65k + ShareGPT4V 35k + DocVQA 7k + charts/spatial-OCR (~200k total) | "mix" (websight/webcode2m/synthetic) + VQA, **2,048** |
| Stage-2 cfg | projector+LoRA **r64/α128/drop0.05**, LR **2e-5**, 1 epoch | LoRA, LR 2e-5 (matches) |
| Compute | **8-GPU DDP** (Prime pod), eff. batch 16×8 | 1× A100, eff. batch 16 |
| Encoder | `siglip-so400m-patch14-384` | `siglip2-so400m-patch16-384` |

**Hyperparameters already match** (LR 1e-3 / 2e-5, resampler/256, LoRA). The gap is **data (scale +
LLaVA-Pretrain alignment)** and **compute (8 GPUs vs 1)**.

## Execution

We execute the recipe in **our own `train.py`**, which already supports both stages — the `align` mix
(reading-biased), the `recon`/`qa` objectives, `--init-projector` warm-start, and `--unfreeze lora`.
No external repo is run; we only borrow the recipe shape (stage split, LRs, LoRA params).

## Plan

1. **Data** — the `align` mix (`corpora.py`) is the Stage-1 set: SyntheticOCR (generated, free) +
   `cauldron_rendered_text` (real rendered-text transcription from `HuggingFaceM4/the_cauldron`) +
   WebSight, reading-dominant. `cauldron_textcaps` / `cauldron_iam` also registered. Next text-rich
   adds: `pixparse/idl-wds` + `pdfa-eng-wds` (real-doc OCR at scale), LLaVAR (OCR-instruction), and
   `the_cauldron` QA configs for Stage-2.
2. **Stage-1 alignment** — `train.py --dataset align --objective recon --unfreeze "" --lr 1e-3`
   (projector-only) → a `best.pt`. Launch: `logs/run_stage1_align.sh`.
3. **Stage-2 instruction** — `train.py --objective qa --unfreeze lora --lr 2e-5 --init-projector <stage1>`
   on the reading/VQA mix → the reader checkpoint.
4. **Evaluate on our fixed yardstick** — `dataset_qa_accuracy` over the seeded val split (fixed,
   commit 41f6297) + the external benchmark suite.
5. **Then** revisit RLVR (GSPO machinery is ready: stabilized, G=16 micro-batching, fair eval) on a
   reader that actually reads — RL amplifies signal that now exists.

## Open decisions (need user input)

- **Compute**: rent a Prime **8×A100/H100 pod** (~hours, matches the 8-GPU recipe) vs a **long
  single-GPU run** on this A100 (200k×1ep ≈ 12.5k steps; feasible but slow, ~half-to-full day). Pod is
  faster + is what the recipe assumes; local is free but serial.
- **Stage-2 budget**: full ~200k (closest to reference) vs a cheaper ~50k first pass to confirm the
  approach lifts qa_acc before committing the full run.
