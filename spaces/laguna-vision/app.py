"""Gradio Space: Laguna XS.2 — a text-only coding model — handling images in a multi-turn chat via
the trained GLM-OCR -> projector -> frozen-LLM adapter. Upload a screenshot/chart, ask about it, follow
up across turns; each turn's image is spliced at an `<image>` marker (vision as a tool observation).

Config (one app, both base precisions — see docs/specs/2026-05-30-hf-space-vision-demo.md):
  LAGUNA_BASE       poolside/Laguna-XS.2  (bf16; set ...-NVFP4 for the quantized base)
  LAGUNA_PROJECTOR  path or HF repo id (downloads `projector.pt`) for the trained projector weights
  LAGUNA_ENCODER    glm_ocr            frozen vision tower
  LAGUNA_POOL       4                  projector token pooling (must match training)
  LAGUNA_MAX_TOKENS 256                reply length cap
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter

BASE = os.environ.get("LAGUNA_BASE", "poolside/Laguna-XS.2")
PROJECTOR = os.environ.get("LAGUNA_PROJECTOR", "chaleong/laguna-xs2-multimodal")
ENCODER = os.environ.get("LAGUNA_ENCODER", "glm_ocr")
POOL = int(os.environ.get("LAGUNA_POOL", "4"))
MAX_TOKENS = int(os.environ.get("LAGUNA_MAX_TOKENS", "256"))


def _resolve_projector(ref: str) -> str:
    """A local path, else a HF model repo to download from: `org/name` (pulls `projector.pt`) or
    `org/name/file.pt` (pulls that file)."""
    if Path(ref).exists():
        return ref
    if ref.endswith(".pt"):
        repo, _, name = ref.rpartition("/")
        return hf_hub_download(repo_id=repo, filename=name)
    return hf_hub_download(repo_id=ref, filename="projector.pt")


def _load_adapter() -> VisualAdapter:
    adapter = VisualAdapter(load_encoder(ENCODER, pool=POOL), BASE)
    state = torch.load(_resolve_projector(PROJECTOR), map_location=adapter.llm.device)
    adapter.projector.load_state_dict(state)
    adapter.eval()
    return adapter


adapter = _load_adapter()  # once, at startup — fine on a dedicated-GPU Space (ZeroGPU would need the
                           # load + respond inside an @spaces.GPU fn + `spaces` in requirements; see README)


def respond(message: dict, turns: list[Turn],
            chat: list[dict]) -> tuple[list[Turn], list[dict], gr.MultimodalTextbox]:
    """One submit: append the new user turn, regenerate the conversation (greedy -> prior replies are
    reproduced deterministically), show the latest reply. Stateless — `turns` is the whole history."""
    # strip any literal <image> the user typed — markers are added per uploaded image below, so a stray
    # one would desync the marker/image count and raise in _embed_multi.
    text = (message.get("text") or "").replace(IMAGE_TOKEN, "").strip()
    images = [Image.open(f).convert("RGB") for f in message.get("files", [])]
    marker = f"{IMAGE_TOKEN}\n" * len(images)  # one <image> per uploaded image, filled by this turn
    turns = turns + [Turn(f"{marker}{text}", images)]

    reply = adapter.chat(turns, max_new_tokens=MAX_TOKENS)[-1]

    chat = chat + [{"role": "user", "content": text or "(image)"},
                   {"role": "assistant", "content": reply}]
    return turns, chat, gr.MultimodalTextbox(value=None)


with gr.Blocks(title="Laguna XS.2 — vision chat") as demo:
    gr.Markdown(f"## Laguna XS.2 sees images\nText-only `{BASE}` + frozen GLM-OCR + trained projector. "
                "Upload a screenshot or chart and ask about it across turns.")
    chatbot = gr.Chatbot(type="messages", height=460)
    turns_state = gr.State([])
    box = gr.MultimodalTextbox(placeholder="Attach an image and ask…", file_types=["image"])
    box.submit(respond, [box, turns_state, chatbot], [turns_state, chatbot, box])

if __name__ == "__main__":
    demo.launch()
