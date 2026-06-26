# Encoder-free VLM path for Laguna (Gemma 4 / Fuyu style) — POC

**Branch:** `feat/encoder-free-vlm` · **Date:** 2026-06-19

Adopt the **encoder-free** VLM design — drop the pretrained vision tower and feed **raw pixel patches** straight into the frozen LLM through one trainable linear embedder. This is the architecture Google DeepMind shipped in **Gemma 4 12B** (encoder-free multimodal, Jun 2026), lineage **Fuyu → EVE → SOLO**; the walkthrough we follow for the embedder recipe is HuggingFaceM4's [*Train Your Own Encoder-Free VLM in $100*](https://huggingface.co/spaces/HuggingFaceM4/encoder-free-vlm). See [References](#references) for the papers.

## Why it drops into Laguna cleanly

`VisualAdapter` is already `encoder → trainable projector → splice at <image>`. The encoder-free embedder is a **swap for the frozen SigLIP/GLM tower**, not a new pipeline:

| Reference embedder step | Where it lands in Laguna |
|---|---|
| resize shorter-side→512, center-crop 512×512, patchify 32px → 256×3072 raw patches | `PatchifyEncoder` (new, ~0 params) — conforms to the `Encoder` interface (`encode → (B,N,d_enc)`, `d_enc`) |
| LayerNorm → Linear(3072→H) → LayerNorm → +factorized row/col pos → LayerNorm → connector | `PatchEmbedder` = new `"patch_embed"` projector kind (the trainable bridge) |
| splice 256 patch vectors at `<image>` placeholders; CE with image/pad masked | unchanged — `_embed_multi` already splices an N-token block at one `<image>` marker |

Because the swap is at the `Encoder`/`Projector` seam, the **entire SFT / GSPO / eval / probe toolchain inherits for free** — `forward`, `forward_qa`, `transcribe`, `chat`, and the GSPO trainer all operate on the projected `(B, Nv, H)` tokens regardless of how they were produced.

## Design decisions

- **Encoder = patchifier, projector = embedder body.** Keeps Laguna's invariant "the only trainable module is the projector" (the patchify step is the cacheable frozen part; `PatchEmbedder` is the recomputable trainable part — so GSPO's `feats`-caching seam still works).
- **Grid read from token count, not wired per-encoder.** `PatchEmbedder` sizes `max_grid`×`H` row/col tables and derives the live grid as `g=√N` at forward, so it serves any square patch grid with no `patch_grid` plumbing and no overloaded-`grid` footgun across encoders.
- **Patchify is strictly 1:1 patch→token** (`pool` pinned to 1; `pool>1` raises) — pooling raw patches would scramble the row-major layout the positional table assumes.
- **Factorized positions** (`E_row[i]+E_col[j]`): 2·g·H params vs g²·H for a full table, matching the reference.

## Validation

`tests/test_encoder_free.py` — patchify row-major correctness, encoder interface (shape/dims/upscale/normalize), embedder shape + factorized positions + grad flow, loud failures (`pool>1`, non-square N), and the end-to-end **overfit-one-batch smoke on Qwen3-0.6B** (the raw-pixel path drives masked-CE down through the frozen decoder → wiring is real).

## Plan

1. **Smoke (Qwen3-0.6B)** — cheap sanity that the embedder learns to *see*, mirroring the reference's "scale the decoder" finding (SmolLM2-135M was too weak).
2. **Ship (Laguna-XS.2)** — the real target: tiny trainable embedder feeding the frozen 33B MoE from raw pixels.

Both run via the existing `train.py` (`--encoder patchify --projector patch_embed --pool 1`); queued behind the GPU watcher to fire when the `wm` run frees the A100.

## Out-of-scope follow-ups

- **Batched encode for fixed-shape encoders.** `_project` loops per-image (needed for variable-N NaFlex); `PatchifyEncoder` emits uniform N and could encode + transfer the whole list in one shot at eval. Training runs `micro_batch=1`, so no waste there — deferred to avoid touching the shared hot path.
- **`Encoder` as a `typing.Protocol`** — the interface is duck-typed by convention; a Protocol would make `encode`/`d_enc`/`pool` explicit.

## References

**Patch embedding (origin).** Dosovitskiy et al., [*An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929) (ViT, ICLR 2021) — the linear patch projection + positional embedding our embedder reuses. ViLT, [*Vision-and-Language Transformer Without Convolution or Region Supervision*](https://arxiv.org/abs/2102.03334) (Kim et al., 2021) — early convolution-/encoder-free patch projection into a transformer.

**Encoder-free VLMs (the lineage we follow — linear patch projection).**
- **Fuyu-8B** (Adept, 2023) — [blog](https://www.adept.ai/blog/fuyu-8b); first decoder-only VLM projecting image patches linearly into the LLM, no vision tower. No paper.
- **EVE** — Diao et al., [*Unveiling Encoder-Free Vision-Language Models*](https://arxiv.org/abs/2406.11832) (NeurIPS 2024). The canonical encoder-free study + training recipe.
- **EVEv2** — Diao et al., [*Improved Baselines for Encoder-Free Vision-Language Models*](https://arxiv.org/abs/2502.06788) (2025). Modality decomposition + scaling.
- **SOLO** — Chen et al., [*A Single Transformer for Scalable Vision-Language Modeling*](https://arxiv.org/abs/2407.06438) (TMLR 2024). Fuyu-style linear projection, reproducible recipe.

**Most relevant to Laguna (visual capacity inside a frozen/MoE backbone).**
- **Mono-InternVL** — Luo et al., [*Pushing the Boundaries of Monolithic Multimodal LLMs with Endogenous Visual Pre-training*](https://arxiv.org/abs/2410.08202) (CVPR 2025) and [**Mono-InternVL-1.5**](https://arxiv.org/abs/2507.12566) (2025) — adds visual experts via a **multimodal MoE** in the LLM; directly germane since Laguna-XS.2 is an MoE backbone (a future step beyond the linear embedder).
- **BREEN** — Li et al., [*Bridge Data-Efficient Encoder-Free Multimodal Learning with Learnable Queries*](https://arxiv.org/abs/2503.12446) (2025). Data-efficient learnable-query bridge.
- **HoVLE** — Tao et al., [*Unleashing the Power of Monolithic VLMs with Holistic Vision-Language Embedding*](https://arxiv.org/abs/2412.16158) (2024).
- **VoRA** — [*Vision as LoRA*](https://arxiv.org/abs/2503.20680) (2025); **Native VL primitives** — [*From Pixels to Words*](https://arxiv.org/abs/2510.14979) (2025).

**The product this POC mirrors.** Google DeepMind, **Gemma 4 12B** — unified encoder-free multimodal (text/image/audio), released 3 Jun 2026: [announcement](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/) · [HF release](https://huggingface.co/blog/gemma4).
