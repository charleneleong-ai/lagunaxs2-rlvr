"""Train the visual-adapter projector (frozen encoder + frozen LLM), realizing an mm_adapter
AdapterPlan TOML. Debug on a small base; point --base at the unquantized Laguna on an 80GB GPU.

  python -m laguna_rlvr.visual.train --base Qwen/Qwen3-0.6B --steps 50
  python -m laguna_rlvr.visual.train --encoder qwen3_vl --base Qwen/Qwen3-0.6B --steps 50
  python -m laguna_rlvr.visual.train --base poolside/Laguna-XS.2   # BF16 backbone, 80GB GPU

The NVFP4/FP8/INT4 Laguna checkpoints store MoE experts as per-expert quantized Linears, which
compressed-tensors can't map onto this modeling revision's fused expert params — the experts load
random. Use the unquantized base; the load-integrity guard in model.py enforces this.
"""
from __future__ import annotations

import argparse
import time
import tomllib
from pathlib import Path

import torch
from autoresearch.gpu_monitor import GPUMonitor
from autoresearch.results import log_experiment
from torch.utils.data import DataLoader

from laguna_rlvr.mm_adapter import plan_from_config, render_plan, validate_a100_40gb
from laguna_rlvr.seed import DEFAULT_SEED, seed_everything
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import VisualAdapter

_DEFAULT_CONFIG = "configs/mm_adapter/a100-40gb-projector.toml"


def _collate(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


def train(config: str = _DEFAULT_CONFIG, encoder: str = "glm_ocr", base: str | None = None,
          steps: int | None = None, n_train: int = 512, pool: int = 4,
          projector_kind: str = "linear", out: str = "results/visual",
          seed: int = DEFAULT_SEED) -> Path:
    seed_everything(seed)
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

    out_dir = Path(out) / f"{encoder}__{Path(base).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "projector.pt"

    # GPUMonitor samples nvidia-smi in the background; always log a results.jsonl row (even on
    # crash/SIGINT) so the sweep tracker sees CRASH instead of a silently-vanished iter.
    t0 = time.monotonic()
    step, last_loss, status = 0, float("nan"), "CRASH"
    monitor = GPUMonitor()
    try:
        with monitor:
            opt.zero_grad()
            while step < max_steps:
                for images, labels in loader:
                    loss = adapter(images, labels).loss / grad_accum
                    loss.backward()
                    if (step + 1) % grad_accum == 0:
                        opt.step()
                        opt.zero_grad()
                    if step % 20 == 0:  # .item() forces a CUDA sync — only on the print cadence
                        print(f"step {step}/{max_steps} loss {loss.item() * grad_accum:.4f}", flush=True)
                    step += 1
                    if step >= max_steps:
                        break
            last_loss = loss.item() * grad_accum  # one final sync for the logged score
        torch.save(adapter.projector.state_dict(), ckpt)
        print(f"saved projector -> {ckpt}", flush=True)
        status = "KEEP"
    finally:
        gpu = monitor.summary()
        print(monitor.format_summary(), flush=True)
        log_experiment(
            tag="mm_adapter", config_name=f"{encoder}__{Path(base).name}",
            score=last_loss, steps=step, status=status,
            description=f"visual projector ({projector_kind}) on {base}",
            runtime_min=(time.monotonic() - t0) / 60,
            extra={"final_loss": last_loss, "backbone": base, "encoder": encoder,
                   "gpu_mean_util_pct": round(gpu.mean_util_pct, 1),
                   "gpu_peak_mem_gb": round(gpu.peak_mem_gb, 1),
                   "gpu_peak_mem_pct": round(gpu.peak_mem_pct, 1)},
        )
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
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = p.parse_args()
    train(a.config, a.encoder, a.base, a.steps, a.n_train, a.pool, a.projector, a.out, a.seed)


if __name__ == "__main__":
    main()
