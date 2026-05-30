"""Train the adapter projector (frozen encoder + frozen LLM), realizing an mm_adapter AdapterPlan
TOML. Modality-agnostic: --modality image (default) reads document images, --modality audio reads
speech. Debug on a small base here; point --base at the NVFP4 Laguna on an 80GB GPU.

  python -m laguna_rlvr.mm.train --base Qwen/Qwen3-0.6B --steps 50
  python -m laguna_rlvr.mm.train --modality audio --base Qwen/Qwen3-0.6B --steps 50 --full
  python -m laguna_rlvr.mm.train          # uses the config's backbone (NVFP4 Laguna)
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import torch
import typer
import wandb
from torch.utils.data import DataLoader, Dataset

from laguna_rlvr.audio.data import LibriSpeechASR
from laguna_rlvr.audio.encoders import load_audio_encoder
from laguna_rlvr.mm.model import ModalityAdapter
from laguna_rlvr.mm.seed import seed_everything
from laguna_rlvr.mm_adapter import plan_from_config, render_plan, validate_a100_40gb
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder

_DEFAULT_CONFIG = "configs/mm_adapter/a100-40gb-projector.toml"
_AUDIO_PROMPT = "Transcribe the speech:"
_DEFAULT_POOL = {"image": 4, "audio": 8}  # audio pools harder: Whisper emits ~1500 frames/30s


def _collate(batch):
    inputs, labels = zip(*batch)
    return list(inputs), list(labels)


def _setup(modality: str, encoder: str | None, pool: int, n_train: int, full: bool) -> tuple[object, Dataset, str | None]:
    """(frozen encoder, dataset, prompt) for a modality. Prompt None -> the default image prompt."""
    if modality == "audio":
        ds = LibriSpeechASR(n=n_train, split="train", source="full" if full else "dummy")
        return load_audio_encoder(encoder or "whisper_small", pool=pool), ds, _AUDIO_PROMPT
    return load_encoder(encoder or "glm_ocr", pool=pool), SyntheticOCR(n=n_train), None


def _val_dataset(modality: str, n: int, full: bool) -> Dataset:
    """Held-out split sharing the train encoder — the live val/loss signal (disjoint from train)."""
    if modality == "audio":
        return LibriSpeechASR(n=n, split="eval", source="full" if full else "dummy")
    return SyntheticOCR(n=n, seed=10_000)  # held out from the training seed (0)


@torch.no_grad()
def _val_loss(adapter: ModalityAdapter, loader: DataLoader) -> float:
    return sum(adapter(inputs, labels).item() for inputs, labels in loader) / len(loader)


def train(config: str = _DEFAULT_CONFIG, modality: str | None = None, encoder: str | None = None,
          base: str | None = None, steps: int | None = None, n_train: int = 512, n_val: int = 16,
          pool: int | None = None, projector_kind: str = "linear", out: str = "results/adapter",
          seed: int = 0, full: bool = False, val_every: int = 100, use_wandb: bool = True) -> Path:
    seed_everything(seed)
    cfg = tomllib.loads(Path(config).read_text())
    plan = plan_from_config(cfg)
    print(render_plan(plan), flush=True)
    training = cfg.get("training", {})
    lr = float(training.get("learning_rate", 1e-4))
    max_steps = steps or int(training.get("max_steps", 1000))
    grad_accum = plan.gradient_accumulation_steps
    base = base or plan.backbone_model
    modality = modality or ("audio" if plan.modality == "audio" else "image")
    pool = pool if pool is not None else _DEFAULT_POOL[modality]

    # Enforce the A100-40GB guardrails only for the configured backbone; a small debug --base is exempt.
    issues = validate_a100_40gb(plan)
    if base == plan.backbone_model and issues:
        raise SystemExit("Guardrail failures for the configured backbone:\n- " + "\n- ".join(issues))

    enc, dataset, prompt = _setup(modality, encoder, pool, n_train, full)
    adapter = ModalityAdapter(enc, base, projector_kind=projector_kind, prompt=prompt)
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=plan.micro_batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_val_dataset(modality, n_val, full), batch_size=plan.micro_batch_size,
                            collate_fn=_collate)

    # Offline when no key (the W&B key lives in .env, sourced into the env, never read here).
    if use_wandb and not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project="laguna-mm-adapter", config={
        "modality": modality, "encoder": encoder or "default", "base": base, "full": full,
        "projector_kind": projector_kind, "lr": lr, "pool": pool, "n_train": n_train,
        "grad_accum": grad_accum, "max_steps": max_steps,
    }) if use_wandb else None

    step = 0
    opt.zero_grad()
    while step < max_steps:
        for inputs, labels in loader:
            loss = adapter(inputs, labels) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad()
            if step % 20 == 0:  # .item() forces a CUDA sync — only at log cadence
                tl = loss.item() * grad_accum
                metrics = {"train/loss": tl}
                if step % val_every == 0:
                    metrics["val/loss"] = _val_loss(adapter, val_loader)
                if run:
                    run.log(metrics, step=step)
                print(f"step {step}/{max_steps} " + " ".join(f"{k} {v:.4f}"
                                                             for k, v in metrics.items()), flush=True)
            step += 1
            if step >= max_steps:
                break

    out_dir = Path(out) / f"{modality}__{encoder or 'default'}__{Path(base).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "projector.pt"
    torch.save(adapter.projector.state_dict(), ckpt)
    print(f"saved projector -> {ckpt}", flush=True)
    if run:
        run.summary["final/val_loss"] = _val_loss(adapter, val_loader)
        print(f"wandb run: {run.url}", flush=True)
        run.finish()
    return ckpt


if __name__ == "__main__":
    typer.run(train)
