---
title: Laguna XS.2 Vision Chat
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
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
| `LAGUNA_PROJECTOR` | `chaleong/laguna-xs2-multimodal` | trained projector — a HF model repo (pulls `projector.pt`), `org/name/file.pt`, or a local path |
| `LAGUNA_ENCODER` / `LAGUNA_POOL` | `glm_ocr` / `4` | frozen vision tower + projector pooling (must match training) |
| `LAGUNA_MAX_TOKENS` | `256` | reply length cap |

**Secret:** `HF_TOKEN` — needed for the gated base **and** the private projector repo.

**Hardware:** bf16 needs a dedicated **A100-80 GB** (~66 GB). NVFP4 (~20 GB) can run on **ZeroGPU**, but
that's not a pure config flip — ZeroGPU only exposes the GPU inside an `@spaces.GPU` function, so you'd
move `_load_adapter()` + `respond` under one and add `spaces` to `requirements.txt`.

## Deploy

The projector is already published to **[`chaleong/laguna-xs2-multimodal`](https://huggingface.co/chaleong/laguna-xs2-multimodal)**
(private; `projector.pt`), which is the `LAGUNA_PROJECTOR` default — so no checkpoint upload is needed.

This folder *is* the Space root (so `app_file: app.py` resolves and `requirements.txt` installs the
`laguna_rlvr` package from git). To publish:

```bash
# 1. create the Space (gradio SDK, dedicated A100) under your account
hf repo create chaleong/laguna-xs2-vision --repo-type space --space-sdk gradio

# 2. set the token secret (gated base + private projector) — value from your gitignored .env
set -a; . .env; set +a   # run from the repo root; loads HF_TOKEN without echoing it
hf repo settings chaleong/laguna-xs2-vision --repo-type space --add-secret HF_TOKEN="$HF_TOKEN"
#   …or paste it in the Space UI: Settings → Variables and secrets

# 3. push THIS folder (its contents at the repo root) to the Space
cd spaces/laguna-vision
git init && git remote add space https://huggingface.co/spaces/chaleong/laguna-xs2-vision
git add . && git commit -m "Laguna XS.2 vision chat" && git push space main

# 4. in the Space settings, set hardware to A100-large (bf16 base is ~66 GB)
```

`requirements.txt` pins the package to the `feat/hf-space-vision` branch — repin to the **merge commit
SHA** before a long-lived public deploy so the build stays reproducible if the branch moves.
