# SFT scale-up plan — close the ~100× data gap to the reference

## Why

The Laguna-XS.2 vision adapter confabulates because the bridge is ~100× under-trained, not because of
RL / encoder / decoding (see session 2026-06-02). `github.com/aaronkazah/laguna-vision` is the **same
architecture** (SigLIP-so400m + AnyRes + resampler→256 tokens + LoRA, frozen Laguna, 2-stage SFT) and
reads — because it runs a proper Stage-1 alignment + Stage-2 instruction tuning at **200k** scale. Ours
(`qasftsigliplora`) ran **2,048** examples and skipped LLaVA-Pretrain alignment entirely.

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

## Two execution paths

- **Path A — run the reference repo directly.** It ships the manifests, `scripts/laguna_llava_stage1.sh`
  / `stage2.sh`, `data/general_recipe.py`, and `prime_*` 8-GPU job scripts. Fastest route to a working
  reader; supply data + compute. Our cleaned-up eval/RL/benchmark infra sits downstream on the resulting
  `projector.pt` (it loads via our `load_adapter_state_dict`). **Recommended.**
- **Path B — port the recipe into our `train.py`.** Add a LLaVA-Pretrain source to `corpora.py` + an
  `align` objective, scale `--n-train`. Keeps our infra, but re-derives what the reference already has.
  More work, more risk.

## Plan (Path A)

1. **Materialize data** — LLaVA-Pretrain 120k (Stage-1) + LLaVA-Instruct/ShareGPT4V/DocVQA + our OCR/
   chart corpora (Stage-2 ~150–200k), via the reference's `data/hf_materialize.py` + `general_recipe.py`.
   COCO train2017 image root needed for the instruct sets.
   - **Wired so far:** the `align` mix now includes `cauldron_rendered_text` (real rendered-text
     transcription from `HuggingFaceM4/the_cauldron`) alongside SyntheticOCR + WebSight, plus
     `cauldron_textcaps` / `cauldron_iam` registered. Next text-rich adds: `pixparse/idl-wds` +
     `pdfa-eng-wds` (real-doc OCR at scale), LLaVAR (OCR-instruction), `the_cauldron` QA configs (Stage-2).
2. **Stage-1 alignment** — `laguna_llava_stage1.sh` (projector-only, LR 1e-3, 1 epoch) → `projector.pt`.
3. **Stage-2 instruction** — `laguna_llava_stage2.sh` warm-started from Stage-1 (projector+LoRA r64/α128,
   LR 2e-5, 1 epoch) → the reader checkpoint.
4. **Evaluate on our fixed yardstick** — `dataset_qa_accuracy` over the seeded val split (already fixed,
   commit 41f6297) + the external benchmark suite. Compare to the reference's ~weak-but-real regime.
5. **Then** revisit RLVR (GSPO machinery is ready: stabilized, G=16 micro-batching, fair eval) on a
   reader that actually reads — RL amplifies signal that now exists.

## Open decisions (need user input)

- **Compute**: rent a Prime **8×A100/H100 pod** (reference's `prime_budget_training_job.sh`, ~hours,
  matches the recipe) vs a **long single-GPU run** on this A100 (200k×1ep ≈ 12.5k steps; feasible but
  slow, ~half-to-full day). Pod is faster + is what the recipe assumes; local is free but serial.
- **Stage-2 budget**: full ~200k (closest to reference) vs a cheaper ~50k first pass to confirm the
  approach lifts qa_acc before committing the full run.
