"""Eval a trained adapter by error rate on a held-out split, vs the zero-training encoder baseline.

  # image (CER vs GLM-OCR-direct floor) — BASE and CKPT are positional:
  python -m laguna_rlvr.mm.eval Qwen/Qwen3-0.6B results/adapter/image__default__Qwen3-0.6B/projector.pt
  # audio (WER vs Whisper-direct reference; --full = real LibriSpeech held-out partition):
  python -m laguna_rlvr.mm.eval Qwen/Qwen3-0.6B results/adapter/audio__whisper_base__Qwen3-0.6B/projector.pt \
      --modality audio --encoder whisper_base --full

Prints one RESULT line: adapter error vs the encoder-direct baseline (GLM-OCR reading the image /
Whisper reading the speech). Whisper is itself a strong ASR model, so its direct WER is a high
reference bar, not a weak floor the projector-into-a-small-LLM path is expected to beat.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import typer
from transformers import (
    AutoImageProcessor,
    AutoModelForImageTextToText,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from laguna_rlvr.audio.data import LibriSpeechASR
from laguna_rlvr.audio.encoders import WHISPER_REPOS, load_audio_encoder
from laguna_rlvr.mm.metrics import cer, wer
from laguna_rlvr.mm.model import ModalityAdapter
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder

_AUDIO_PROMPT = "Transcribe the speech:"
_SAMPLE_RATE = 16_000


def _norm(s: str) -> str:
    """Lowercase + strip punctuation so casing/punctuation don't inflate WER — LibriSpeech refs are
    upper-case unpunctuated while Whisper emits mixed-case punctuated text."""
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _adapter_error(adapter: ModalityAdapter, ds, metric, norm: bool) -> float:
    def score(x, label):
        pred = adapter.transcribe([x])[0]
        return metric(_norm(pred), _norm(label)) if norm else metric(pred, label)

    return sum(score(x, label) for x, label in ds) / len(ds)


@torch.no_grad()
def _glm_baseline(ds, device: str) -> float:
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


@torch.no_grad()
def _whisper_baseline(ds, device: str, repo: str) -> float:
    """Zero-training reference: Whisper transcribes the speech directly (a strong ASR model)."""
    model = WhisperForConditionalGeneration.from_pretrained(repo).eval().to(device)
    proc = WhisperProcessor.from_pretrained(repo)
    scores = []
    for wav, label in ds:
        feats = proc.feature_extractor([wav], sampling_rate=_SAMPLE_RATE, return_tensors="pt")
        gen = model.generate(feats["input_features"].to(device), max_new_tokens=64)
        scores.append(wer(_norm(proc.batch_decode(gen, skip_special_tokens=True)[0]), _norm(label)))
    return sum(scores) / len(scores)


def evaluate(base: str, ckpt: str, modality: str = "image", encoder: str | None = None,
             n_eval: int = 64, pool: int | None = None, projector_kind: str = "linear",
             baseline: bool = True, full: bool = False) -> dict:
    if modality == "audio":
        name = encoder or "whisper_small"
        enc = load_audio_encoder(name, pool=pool or 8)
        ds = LibriSpeechASR(n=n_eval, split="eval", source="full" if full else "dummy")
        metric, metric_name, prompt = wer, "wer", _AUDIO_PROMPT
        base_name, base_fn = "whisper_baseline_wer", lambda dev: _whisper_baseline(ds, dev, WHISPER_REPOS[name])
    else:
        name = encoder or "glm_ocr"
        enc = load_encoder(name, pool=pool or 4)
        ds = SyntheticOCR(n=n_eval, seed=10_000)  # held out from the training seed (0)
        metric, metric_name, prompt = cer, "cer", None
        base_name, base_fn = "glm_baseline_cer", lambda dev: _glm_baseline(ds, dev)

    adapter = ModalityAdapter(enc, base, projector_kind=projector_kind, prompt=prompt)
    adapter.projector.load_state_dict(torch.load(ckpt, map_location=adapter.llm.device))
    result = {"modality": modality, "encoder": name, "base": Path(base).name,
              f"adapter_{metric_name}": _adapter_error(adapter, ds, metric, norm=modality == "audio")}
    if baseline:
        result[base_name] = base_fn(adapter.device)
    print("RESULT " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in result.items()), flush=True)
    return result


if __name__ == "__main__":
    typer.run(evaluate)
