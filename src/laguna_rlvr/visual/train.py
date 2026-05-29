"""Train the visual-adapter projector (frozen encoder + frozen LLM), realizing an mm_adapter
AdapterPlan TOML. Debug on a small base here; point --base at the NVFP4 Laguna on an 80GB GPU.

  python -m laguna_rlvr.visual.train --base Qwen/Qwen3-0.6B --steps 50
  python -m laguna_rlvr.visual.train --encoder qwen3_vl --base Qwen/Qwen3-0.6B --steps 50
  python -m laguna_rlvr.visual.train          # uses the config's backbone (NVFP4 Laguna)
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from laguna_rlvr.mm_adapter import plan_from_config, render_plan, validate_a100_40gb
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import VisualAdapter

_DEFAULT_CONFIG = "configs/mm_adapter/a100-40gb-projector.toml"


def _collate(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


def train(config: str = _DEFAULT_CONFIG, encoder: str = "glm_ocr", base: str | None = None,
          steps: int | None = None, n_train: int = 512, pool: int = 4,
          projector_kind: str = "linear", out: str = "results/visual") -> Path:
    cfg = tomllib.loads(Path(config).read_text())
    plan = plan_from_config(cfg)
    print(render_plan(plan), flush=True)
    training = cfg.get("training", {})
    lr = float(training.get("learning_rate", 1e-4))
    max_steps = steps or int(training.get("max_steps", 1000))
    grad_accum = plan.gradient_accumulation_steps
    base = base or plan.backbone_model

    # Enforce the A100-40GB guardrails only for the configured backbone; a small debug --base is exempt.
    issues = validate_a100_40gb(plan)
    if base == plan.backbone_model and issues:
        raise SystemExit("Guardrail failures for the configured backbone:\n- " + "\n- ".join(issues))

    enc = load_encoder(encoder, pool=pool)
    adapter = VisualAdapter(enc, base, projector_kind=projector_kind)
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr)
    loader = DataLoader(SyntheticOCR(n=n_train), batch_size=plan.micro_batch_size,
                        shuffle=True, collate_fn=_collate)

    step = 0
    opt.zero_grad()
    while step < max_steps:
        for images, labels in loader:
            loss = adapter(images, labels).loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad()
            if step % 20 == 0:
                print(f"step {step}/{max_steps} loss {loss.item() * grad_accum:.4f}", flush=True)
            step += 1
            if step >= max_steps:
                break

    out_dir = Path(out) / f"{encoder}__{Path(base).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "projector.pt"
    torch.save(adapter.projector.state_dict(), ckpt)
    print(f"saved projector -> {ckpt}", flush=True)
    return ckpt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--encoder", default="glm_ocr", choices=["glm_ocr", "qwen3_vl"])
    p.add_argument("--base", default=None, help="override the config backbone (e.g. Qwen/Qwen3-0.6B for debug)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--projector", default="linear", choices=["linear", "mlp"])
    p.add_argument("--out", default="results/visual")
    a = p.parse_args()
    train(a.config, a.encoder, a.base, a.steps, a.n_train, a.pool, a.projector, a.out)


if __name__ == "__main__":
    main()
