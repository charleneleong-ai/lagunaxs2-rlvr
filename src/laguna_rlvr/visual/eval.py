"""Eval the trained visual adapter by CER, alongside the zero-training GLM-OCR baseline floor.

  python -m laguna_rlvr.visual.eval --base Qwen/Qwen3-0.6B --ckpt results/visual/glm_ocr__Qwen3-0.6B/projector.pt

Prints one RESULT line: adapter CER (the projector's reading through the base LLM) vs
glm_baseline CER (GLM-OCR reading the image directly — the no-adapter floor the adapter must beat).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageTextToText

from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.metrics import cer, generation_metrics
from laguna_rlvr.visual.model import VisualAdapter

_DEFAULT_CONFIG = "configs/mm_adapter/a100-40gb-projector.toml"


def _adapter_cer(adapter: VisualAdapter, ds: SyntheticOCR) -> float:
    return generation_metrics(adapter, list(ds), prefix="adapter")["adapter/metrics/cer"]


@torch.no_grad()
def _glm_baseline_cer(ds: SyntheticOCR, device: str) -> float:
    """Zero-training floor: GLM-OCR reads the image directly (image -> OCR -> text)."""
    repo = "zai-org/GLM-OCR"
    model = AutoModelForImageTextToText.from_pretrained(repo, device_map=device).eval()
    proc = AutoImageProcessor.from_pretrained(repo)
    scores = []
    for img, label in ds:
        batch = proc(images=[img], return_tensors="pt").to(model.device)
        gen = model.generate(**batch, max_new_tokens=48, do_sample=False)
        scores.append(cer(proc.batch_decode(gen, skip_special_tokens=True)[0], label))
    return sum(scores) / len(scores)


def evaluate(encoder: str, base: str, ckpt: str, n_eval: int, pool: int,
             projector_kind: str, baseline: bool) -> dict:
    ds = SyntheticOCR(n=n_eval, seed=10_000)  # held out from the training seed (0)
    enc = load_encoder(encoder, pool=pool)
    adapter = VisualAdapter(enc, base, projector_kind=projector_kind)
    adapter.projector.load_state_dict(torch.load(ckpt, map_location=adapter.llm.device))
    result = {"encoder": encoder, "base": Path(base).name, "adapter_cer": _adapter_cer(adapter, ds)}
    if baseline:
        result["glm_baseline_cer"] = _glm_baseline_cer(ds, adapter.device)
    print("RESULT " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in result.items()), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", default="glm_ocr", choices=["glm_ocr", "qwen3_vl"])
    p.add_argument("--base", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--projector", default="linear", choices=["linear", "mlp"])
    p.add_argument("--no-baseline", action="store_true", help="skip the GLM-OCR baseline floor")
    a = p.parse_args()
    evaluate(a.encoder, a.base, a.ckpt, a.n_eval, a.pool, a.projector, not a.no_baseline)


if __name__ == "__main__":
    main()
