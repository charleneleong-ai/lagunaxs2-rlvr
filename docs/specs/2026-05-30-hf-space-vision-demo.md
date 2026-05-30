# HF Space — Laguna XS.2 vision chat demo

**Status:** design · **Branch:** `feat/hf-space-vision` · **Date:** 2026-05-30

## Goal

A public Gradio Space where Laguna XS.2 — text-only by itself — **handles images inside a multi-turn
conversation** via the trained GLM-OCR→projector→frozen-LLM adapter. Upload a screenshot/chart, ask
about it, follow up across turns; vision arrives as a spliced `<image>` observation, not a fixed prefix.

## What already exists (no new model code)

- `VisualAdapter.chat(turns: list[Turn])` — multi-turn multimodal inference. Each `Turn(text, images)`
  carries `<image>` markers filled by that turn's images. **Greedy** (`do_sample=False`), so replaying
  the full user-turn list deterministically reproduces prior replies + appends the new one — the app
  stays stateless (no history-embedding injection needed).
- `load_causal_lm` is quantization-aware → the same app serves **bf16** or **NVFP4** by config alone.
- Trained projectors: `glm_ocr__Laguna-XS.2__mix/best.pt` (bf16) and `…-NVFP4/projector.pt` (NVFP4).

## App (`spaces/laguna-vision/`)

- `app.py` — `gr.Blocks` + `gr.MultimodalTextbox` (text + image upload) + `gr.Chatbot`. A `gr.State`
  holds `list[Turn]`; each submit appends `Turn("<image>\n…"*k + text, imgs)`, calls `adapter.chat`,
  shows `replies[-1]`. Model loaded once at startup.
- Config by env (one app, both precisions):
  | var | default | meaning |
  |---|---|---|
  | `LAGUNA_BASE` | `poolside/Laguna-XS.2` | bf16; set `…-NVFP4` for the quantized base |
  | `LAGUNA_PROJECTOR` | local `best.pt` path | path, or a HF repo id to `hf_hub_download` `projector.pt` |
  | `LAGUNA_ENCODER` / `LAGUNA_POOL` | `glm_ocr` / `4` | frozen vision tower + projector pooling |
- `requirements.txt` — installs the `laguna_rlvr` package + torch/transformers/compressed-tensors/
  tiktoken/sentencepiece/gradio/huggingface_hub.
- `README.md` — Space metadata (`sdk: gradio`, `app_file: app.py`, hardware tag). `HF_TOKEN` secret for
  the gated base.

## Hosting

Primary = **bf16 on a dedicated A100-80 GB Space** (best quality, matches the Stage-0 baseline; ~66 GB
fits). NVFP4 is a config swap for a lighter/ZeroGPU always-on demo (add `@spaces.GPU` for ZeroGPU).

## Out of scope

- Token streaming (greedy full-reply is fine for a demo); O(n²) full-context replay is acceptable for
  short chats — cap turns / trim if it bites.
- Hosting the base weights ourselves (pulled from the hub at runtime).
