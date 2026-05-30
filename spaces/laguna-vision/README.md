---
title: Laguna XS.2 Vision Chat
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
hardware: a100-large
---

# Laguna XS.2 — vision chat

A text-only coding model (`poolside/Laguna-XS.2`, 33 B MoE) **handling images inside a multi-turn
conversation** through a trained GLM-OCR → projector → frozen-LLM adapter. Upload a screenshot or chart
and ask about it; vision is spliced at an `<image>` marker each turn (a tool observation, not a prefix).

## Configuration (env / Space variables)

| var | default | meaning |
|---|---|---|
| `LAGUNA_BASE` | `poolside/Laguna-XS.2` | bf16 base; set `poolside/Laguna-XS.2-NVFP4` for the lighter quantized base |
| `LAGUNA_PROJECTOR` | `results/visual/glm_ocr__Laguna-XS.2__mix/best.pt` | trained projector — a path, or a HF repo id whose `projector.pt` is downloaded |
| `LAGUNA_ENCODER` / `LAGUNA_POOL` | `glm_ocr` / `4` | frozen vision tower + projector pooling (must match training) |
| `LAGUNA_MAX_TOKENS` | `256` | reply length cap |

**Secret:** `HF_TOKEN` — the base is gated.

**Hardware:** bf16 needs a dedicated **A100-80 GB** (~66 GB). For NVFP4 (~20 GB) on ZeroGPU, set
`LAGUNA_BASE`/`LAGUNA_PROJECTOR` to the NVFP4 pair and wrap `respond` in `@spaces.GPU`.

## Deploy

The projector must be reachable by the Space — upload `best.pt` to a HF model repo and point
`LAGUNA_PROJECTOR` at it (the default path only works for local runs on the training machine). Then push
this folder to the Space's git remote.
