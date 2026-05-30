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

import os
import time
import tomllib
from pathlib import Path

import torch
import typer
import wandb
from autoresearch.gpu_monitor import GPUMonitor
from autoresearch.results import log_experiment
from torch.utils.data import DataLoader, Dataset

from laguna_rlvr.mm_adapter import plan_from_config, render_plan, validate_gpu_budget
from laguna_rlvr.seed import DEFAULT_SEED, seed_everything
from laguna_rlvr.visual.corpora import REGISTRY, build_corpus
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import VisualAdapter

_DEFAULT_CONFIG = "configs/mm_adapter/a100-80gb-laguna-bf16.toml"


def _collate(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


@torch.no_grad()
def _val_loss(adapter: VisualAdapter, loader: DataLoader) -> float:
    """Mean per-example loss over the val set — measures generalization, not memorization."""
    adapter.llm.gradient_checkpointing_disable()  # no backward in eval -> checkpointing is pure overhead
    try:
        total, n = 0.0, 0
        for images, labels in loader:
            total += adapter(images, labels).loss.item() * len(labels)
            n += len(labels)
    finally:
        adapter.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return total / max(n, 1)


def _log_samples(run, ds: Dataset, key: str, n: int = 8) -> None:
    """Log a few (image, text) pairs as a W&B table so the corpus is inspectable in the dashboard."""
    table = wandb.Table(columns=["image", "text"])
    for i in range(min(n, len(ds))):
        img, txt = ds[i]
        table.add_data(wandb.Image(img), txt)
    run.log({key: table}, step=0)


def _save_resume(path: Path, adapter: VisualAdapter, opt, step: int, run) -> None:
    """Atomically write resume state (projector + optimizer + step + W&B id) for crash recovery."""
    tmp = path.with_suffix(".tmp")
    torch.save({"projector": adapter.projector.state_dict(), "opt": opt.state_dict(),
                "step": step, "wandb_id": run.id if run else None}, tmp)
    tmp.replace(path)  # rename is atomic — a crash mid-write never corrupts the live checkpoint


def train(config: str = _DEFAULT_CONFIG, encoder: str = "glm_ocr", base: str | None = None,
          steps: int | None = None, n_train: int = 512, pool: int = 4,
          projector_kind: str = "linear", out: str = "results/visual",
          seed: int = DEFAULT_SEED, dataset: str = "synthetic", use_wandb: bool = True,
          resume: bool = True) -> Path:
    seed_everything(seed)
    cfg = tomllib.loads(Path(config).read_text())
    plan = plan_from_config(cfg)
    print(render_plan(plan), flush=True)
    training = cfg.get("training", {})
    lr = float(training.get("learning_rate", 1e-4))
    max_steps = steps or int(training.get("max_steps", 1000))
    grad_accum = plan.gradient_accumulation_steps
    base = base or plan.backbone_model

    # Enforce the VRAM-budget guardrails only for the configured backbone; a small debug --base is exempt.
    issues = validate_gpu_budget(plan)
    if base == plan.backbone_model and issues:
        raise SystemExit("Guardrail failures for the configured backbone:\n- " + "\n- ".join(issues))

    enc = load_encoder(encoder, pool=pool)
    adapter = VisualAdapter(enc, base, projector_kind=projector_kind)
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr)

    full = build_corpus(dataset, n_train)
    n_val = max(1, len(full) // 10)  # 90/10 split, seeded for reproducibility
    train_ds, val_ds = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_ds, batch_size=plan.micro_batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=plan.micro_batch_size, shuffle=False, collate_fn=_collate)
    val_every = max(20, max_steps // 10)

    out_dir = Path(out) / f"{encoder}__{Path(base).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "projector.pt"

    # Resume a crashed/pre-empted run: restore projector + optimizer + step, and rejoin the W&B run.
    resume_ckpt = out_dir / "resume.pt"
    start_step, resume_id = 0, None
    if resume and resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location="cpu")  # load_state_dict places opt state on the param device
        adapter.projector.load_state_dict(state["projector"])
        opt.load_state_dict(state["opt"])
        start_step, resume_id = state["step"], state.get("wandb_id")
        print(f"resuming from step {start_step}/{max_steps}", flush=True)

    # Offline when no WANDB_API_KEY (still produces a local trace to sync later); online otherwise.
    run = None
    if use_wandb:
        if not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")
        run = wandb.init(project="laguna-mm-adapter", name=f"{encoder}__{Path(base).name}",
                         id=resume_id, resume="allow" if resume_id else None,
                         config={"base": base, "encoder": encoder, "projector": projector_kind,
                                 "dataset": dataset, "lr": lr, "max_steps": max_steps,
                                 "grad_accum": grad_accum, "n_train": n_train, "seed": seed})
        if start_step == 0:  # sample tables log at step 0 — skip when rejoining a resumed run
            _log_samples(run, train_ds, "train/samples")
            _log_samples(run, val_ds, "val/samples")

    # GPUMonitor samples nvidia-smi in the background; always log a results.jsonl row (even on
    # crash/SIGINT) so the sweep tracker sees CRASH instead of a silently-vanished iter.
    t0 = time.monotonic()
    step, last_loss, last_val, status = start_step, float("nan"), float("nan"), "CRASH"
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
                    if step % 20 == 0:  # .item() forces a CUDA sync — only on the log cadence
                        cur = loss.item() * grad_accum
                        print(f"step {step}/{max_steps} loss {cur:.4f}", flush=True)
                        if run:
                            run.log({"train/loss": cur}, step=step)
                    if step % val_every == 0:
                        last_val = _val_loss(adapter, val_loader)
                        print(f"  val loss {last_val:.4f}", flush=True)
                        if run:
                            run.log({"val/loss": last_val}, step=step)
                    step += 1
                    if step % val_every == 0:  # crash-recovery checkpoint at the val cadence
                        _save_resume(resume_ckpt, adapter, opt, step, run)
                    if step >= max_steps:
                        break
            last_loss = loss.item() * grad_accum  # one final sync for the logged score
            last_val = _val_loss(adapter, val_loader)
            if run:
                run.log({"val/loss": last_val}, step=step)
        torch.save(adapter.projector.state_dict(), ckpt)
        print(f"saved projector -> {ckpt} (train {last_loss:.4f}, val {last_val:.4f})", flush=True)
        status = "KEEP"
        resume_ckpt.unlink(missing_ok=True)  # completed — don't let a fresh run resume it
    finally:
        gpu = monitor.summary()
        print(monitor.format_summary(), flush=True)
        if run:
            run.summary.update({"final_loss": last_loss, "val_loss": last_val, "status": status,
                                "gpu_peak_mem_gb": gpu.peak_mem_gb,
                                "gpu_mean_util_pct": gpu.mean_util_pct})
            run.finish()
        log_experiment(
            tag="mm_adapter", config_name=f"{encoder}__{Path(base).name}",
            score=last_loss, steps=step, status=status,
            description=f"visual projector ({projector_kind}) on {base} [{dataset}]",
            wandb_url=run.url if run else "",
            runtime_min=(time.monotonic() - t0) / 60,
            extra={"final_loss": last_loss, "val_loss": last_val, "backbone": base,
                   "encoder": encoder, "dataset": dataset,
                   "gpu_mean_util_pct": round(gpu.mean_util_pct, 1),
                   "gpu_peak_mem_gb": round(gpu.peak_mem_gb, 1),
                   "gpu_peak_mem_pct": round(gpu.peak_mem_pct, 1)},
        )
    return ckpt


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: str = _DEFAULT_CONFIG,
    encoder: str = typer.Option("glm_ocr", help="glm_ocr | qwen3_vl"),
    base: str = typer.Option(None, help="override the config backbone (e.g. Qwen/Qwen3-0.6B for debug)"),
    steps: int = typer.Option(None, help="override config max_steps"),
    n_train: int = 512,
    pool: int = 4,
    projector: str = typer.Option("linear", help="linear | mlp"),
    out: str = "results/visual",
    seed: int = DEFAULT_SEED,
    dataset: str = typer.Option("synthetic", help=" | ".join(REGISTRY)),
    wandb_tracking: bool = typer.Option(True, "--wandb/--no-wandb", help="log to Weights & Biases"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="resume from resume.pt if present"),
) -> None:
    train(config, encoder, base, steps, n_train, pool, projector, out, seed, dataset, wandb_tracking,
          resume)


if __name__ == "__main__":
    app()
