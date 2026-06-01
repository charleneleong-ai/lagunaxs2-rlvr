"""Minimal local HTTP server for the visual adapter — POST {image, question}, get the answer back.

The simplest way to serve the adapter on an 80GB GPU: load `VisualAdapter` once, then `adapter.chat`
per request (HF `generate(inputs_embeds=...)`, vision spliced at `<image>`). No vLLM, no gateway — for
fast batched serving use the `--enable-prompt-embeds` vLLM gateway (laguna-vision-vllm) instead; this is
the dependency-free path for trying / demoing a checkpoint.

  python -m laguna_rlvr.visual.serve --ckpt results/visual/<run>/best.pt --encoder siglip --unfreeze lora
  curl -s localhost:8100 -d '{"image":"https://.../x.jpg","question":"What does it say?"}'
"""
from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import torch
import typer
from PIL import Image

from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter

app = typer.Typer(add_completion=False, help=__doc__)


def _fetch(image: str) -> Image.Image:
    """Load an image from a URL or a local path."""
    raw = httpx.get(image, timeout=30).content if image.startswith("http") else open(image, "rb").read()
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.command()
def main(ckpt: str = typer.Option(..., help="adapter checkpoint (projector + LoRA)"),
         encoder: str = "siglip", base: str = "poolside/Laguna-XS.2", projector: str = "resampler",
         unfreeze: str = "lora", host: str = "0.0.0.0", port: int = 8100,
         max_new_tokens: int = 128) -> None:
    """Serve the adapter: POST JSON {image: <url|path>, question: str, max_new_tokens?: int} -> {answer}."""
    adapter = VisualAdapter(load_encoder(encoder, pool=(4 if "qwen" in encoder else 1)), base,
                            projector_kind=projector, use_anchor=False, unfreeze=unfreeze)
    adapter.load_adapter_state_dict(torch.load(ckpt, map_location=adapter.llm.device))
    print(f"loaded adapter from {ckpt}; serving on {host}:{port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                reply = adapter.chat([Turn(f"{IMAGE_TOKEN}\n{req.get('question', 'Describe the image.')}",
                                           [_fetch(req["image"])])],
                                     max_new_tokens=req.get("max_new_tokens", max_new_tokens))[0]
                payload, code = {"answer": reply}, 200
            except Exception as e:  # a bad request shouldn't kill the server
                payload, code = {"error": f"{type(e).__name__}: {e}"}, 400
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):  # quiet — the adapter print is the only signal we want
            pass

    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    app()
