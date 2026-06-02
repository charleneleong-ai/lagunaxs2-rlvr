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
from laguna_rlvr.visual.corpora import (CHOICES, DEFAULT_VQA, QASFTDataset, build_corpus, load_vqa,
                                        parse_mixture, read_question)
from laguna_rlvr.visual.multiturn_qa import (_RECALL_Q, dataset_qa_accuracy, evaluate_multiturn_qa,
                                             image_fetcher, mixture_episodes)
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.hf_image_text import HFImageTextDataset
from laguna_rlvr.visual.metrics import generation_metrics
from laguna_rlvr.visual.model import VisualAdapter

_DEFAULT_CONFIG = "configs/mm_adapter/a100-80gb-laguna-bf16.toml"


def _collate(batch):
    cols = list(zip(*batch))
    images, labels = list(cols[0]), list(cols[1])
    corpora = list(cols[2]) if len(cols) > 2 else [None] * len(images)  # corpus tag (mix) or None
    questions = list(cols[3]) if len(cols) > 3 else [None] * len(images)  # per-example QA question (VQA)
    return images, labels, corpora, questions


def _ocr_probe(seed: int, n: int = 32) -> list:
    """Held-out general-OCR retention probe: real document pages (moondream/ia_ocr) → text. GLM-OCR's
    home turf — scored in val + eval so we can see the projector keeps the frozen base's native OCR
    ability, not just the screenshot→code mix. ia_ocr is not a training corpus, so any slice is held
    out; text is capped short to match the 48-token transcribe budget. Falls back to the in-process
    SyntheticOCR floor if the remote set can't be fetched, so a detached run can't die on a download.
    """
    try:
        return list(HFImageTextDataset("moondream/ia_ocr", n=n, max_text_chars=200))
    except Exception as e:  # network / schema / cache failure -> degrade to the floor, don't crash
        print(f"  [ocr-probe] moondream/ia_ocr unavailable ({type(e).__name__}: {e}); "
              "using SyntheticOCR floor", flush=True)
        return list(SyntheticOCR(n=16, seed=seed))


def _loss(adapter: VisualAdapter, images, labels, corpora, objective: str, questions=None):
    """Dispatch the training loss by objective: reconstruction (transcribe) vs QA-SFT (answer the
    per-kind question — forces vision use, since the needle answer isn't in the question)."""
    if objective == "qa":
        return adapter.forward_qa(images, labels, corpora, questions).loss
    return adapter(images, labels).loss


@torch.no_grad()
def _val_loss(adapter: VisualAdapter, loader: DataLoader, objective: str = "recon") -> tuple[float, dict[str, float]]:
    """Mean per-example val loss, plus a per-corpus breakdown (so swe-vs-websight ranges are visible)."""
    adapter.llm.gradient_checkpointing_disable()  # no backward in eval -> checkpointing is pure overhead
    try:
        total, n = 0.0, 0
        per_sum: dict[str, float] = {}
        per_n: dict[str, int] = {}
        for images, labels, corpora, questions in loader:
            li = _loss(adapter, images, labels, corpora, objective, questions).item() * len(labels)
            total += li
            n += len(labels)
            c = corpora[0]
            if c is not None:
                per_sum[c] = per_sum.get(c, 0.0) + li
                per_n[c] = per_n.get(c, 0) + len(labels)
    finally:
        adapter.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return total / max(n, 1), {c: per_sum[c] / per_n[c] for c in per_sum}


def _log_samples(run, ds: Dataset, key: str, n: int = 8) -> None:
    """Log a few (image, text) pairs as a W&B table so the corpus is inspectable in the dashboard."""
    table = wandb.Table(columns=["image", "text"])
    for i in range(min(n, len(ds))):
        item = ds[i]
        table.add_data(wandb.Image(item[0]), item[1])  # item may be (img, txt) or (img, txt, corpus)
    run.log({key: table}, step=0)


def _log_qa_samples(run, n: int = 4) -> None:
    """Log example multi-turn multimodal QA episodes (the target task: read A, read B, recall A) as a
    W&B table, so the qa/metrics/* probe is inspectable next to its scores."""
    eps = mixture_episodes(n, per_corpus=16)
    if not eps:
        return
    fetch = image_fetcher(eps)
    table = wandb.Table(columns=["image_A", "Q1 read A", "needle_A",
                                 "image_B", "Q2 read B", "needle_B", "Q3 recall", "expected"])
    for ep in eps:
        table.add_data(wandb.Image(fetch(ep.a)), read_question(ep.a.kind), ep.a.needle,
                       wandb.Image(fetch(ep.b)), read_question(ep.b.kind), ep.b.needle, _RECALL_Q, ep.a.needle)
    run.log({"qa/samples": table}, step=0)


def _log_predictions(run, adapter, items: list, key: str, step: int, n: int = 8) -> None:
    """Log (image, target, prediction) for a few items so what the adapter actually READS is inspectable
    next to the val/eval scores as training progresses (the input corpus is logged once at step 0; this
    is the model's evolving output)."""
    sample = items[:n]
    preds = adapter.transcribe([it[0] for it in sample])
    table = wandb.Table(columns=["image", "target", "prediction"])
    for it, p in zip(sample, preds):
        table.add_data(wandb.Image(it[0]), str(it[1])[:200], str(p)[:200])
    run.log({key: table}, step=step)


def _save_resume(path: Path, adapter: VisualAdapter, opt, step: int, run) -> None:
    """Atomically write resume state (trainable adapter + optimizer + step + W&B id) for crash recovery."""
    tmp = path.with_suffix(".tmp")
    torch.save({"adapter": adapter.adapter_state_dict(), "opt": opt.state_dict(),
                "step": step, "wandb_id": run.id if run else None}, tmp)
    tmp.replace(path)  # rename is atomic — a crash mid-write never corrupts the live checkpoint


def train(config: str = _DEFAULT_CONFIG, encoder: str = "glm_ocr", base: str | None = None,
          steps: int | None = None, n_train: int = 512, pool: int = 4,
          projector_kind: str = "linear", out: str = "results/visual",
          seed: int = DEFAULT_SEED, dataset: str = "synthetic", use_wandb: bool = True,
          resume: bool = True, mixture: str = "", name_suffix: str = "",
          eval_dataset: str = "", patience: int = 3, min_delta: float = 1e-3,
          qa_eval: bool = True, description: str = "", init_projector: str = "",
          objective: str = "recon", unfreeze: str = "", use_anchor: bool = True,
          lr_override: float | None = None, vqa: str = "default", norm_penalty: float = 0.0) -> Path:
    seed_everything(seed)
    cfg = tomllib.loads(Path(config).read_text())
    plan = plan_from_config(cfg)
    print(render_plan(plan), flush=True)
    training = cfg.get("training", {})
    lr = lr_override or float(training.get("learning_rate", 1e-4))
    max_steps = steps or int(training.get("max_steps", 1000))
    grad_accum = plan.gradient_accumulation_steps
    base = base or plan.backbone_model

    # Enforce the VRAM-budget guardrails only for the configured backbone; a small debug --base is exempt.
    issues = validate_gpu_budget(plan)
    if base == plan.backbone_model and issues:
        raise SystemExit("Guardrail failures for the configured backbone:\n- " + "\n- ".join(issues))

    enc = load_encoder(encoder, pool=pool)
    adapter = VisualAdapter(enc, base, projector_kind=projector_kind, unfreeze=unfreeze,
                            use_anchor=use_anchor, norm_penalty=norm_penalty)
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr)

    # Mixture weights: explicit --mixture always wins. Otherwise the config's [mixture].weights is the
    # prior for `--dataset mix` ONLY — `align` (and any other named dataset) carries its own built-in
    # mix (_ALIGN_MIX), so the config's mix weights must NOT bleed into it (else --dataset align silently
    # runs the code-heavy default mix and erodes readout — caught 2026-06-02).
    if mixture:
        mix_specs = parse_mixture(mixture)
    elif dataset == "mix":
        mix_specs = [(k, float(v)) for k, v in cfg.get("mixture", {}).get("weights", {}).items()] or None
    else:
        mix_specs = None
    full = build_corpus(dataset, n_train, mixture=mix_specs)
    if objective == "qa":  # QA-SFT: (image, answer, corpus, question) from needle rows + VQA reading sets
        vqa_names = DEFAULT_VQA if vqa == "default" else [s for s in vqa.split(",") if s]
        full = QASFTDataset(full, vqa_sources=load_vqa(vqa_names, n_train) if vqa_names else None)
    n_val = min(max(1, len(full) // 10), 256)  # 90/10 split, capped so frequent val stays cheap
    train_ds, val_ds = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    # Parallel per-step data loading so the GPU isn't fed single-process (util starves otherwise). The
    # heavy vision-encode runs in the train step (GPU-side), so workers mainly parallelize PIL decode +
    # collate -> modest but real. persistent_workers keeps them alive across the frequent val/eval passes.
    dl = dict(batch_size=plan.micro_batch_size, collate_fn=_collate, num_workers=4,
              pin_memory=True, persistent_workers=True)
    loader = DataLoader(train_ds, shuffle=True, **dl)
    val_loader = DataLoader(val_ds, shuffle=False, **dl)
    eval_loader = DataLoader(build_corpus(eval_dataset, 128), shuffle=False, **dl) if eval_dataset else None
    val_every = max(50, min(max_steps // 10, 500))  # loss/early-stop cadence (bounded for high ceilings)
    gen_every = val_every * 3  # generation metrics (WER/CER) are slow -> coarser cadence than loss val
    wer_items = [val_ds[i] for i in range(min(16, len(val_ds)))]  # fixed subset for WER/CER (generation)
    # fixed val sample for full-distribution QA-read accuracy (incl. VQA/synthetic, per corpus)
    qa_eval_items = [val_ds[i] for i in range(min(40, len(val_ds)))] if objective == "qa" else []
    # Inline reading probe for non-QA objectives (recon Stage-1): a fixed REAL-VQA sample, independent
    # of the (synthetic-heavy) train mix, so we watch real-image reading TRANSFER live each eval rather
    # than stop-and-eval. Observational only — logged under read/, never drives val selection. Built
    # defensively: a VQA fetch failure must not kill a detached run (cf. _ocr_probe).
    read_probe_items: list = []
    if qa_eval and objective != "qa":
        try:
            _probe = QASFTDataset([], vqa_sources=load_vqa(DEFAULT_VQA, 5))
            read_probe_items = [_probe[i] for i in range(min(24, len(_probe)))]
        except Exception as e:  # noqa: BLE001 — never let a probe-data fetch kill training
            print(f"read-probe build failed ({e}); skipping inline reading probe", flush=True)
    wer_images = [it[0] for it in wer_items]  # same subset's images, reused for the embedding-drift gauge
    ocr_probe = _ocr_probe(seed=10_007)  # held-out general-OCR retention probe (val + eval)

    # Scope the run identity by dataset too — otherwise different corpora on the same backbone share
    # out_dir/resume.pt and the W&B run, so a crashed run is wrongly resumed by the next (caught 2026-05).
    run_name = f"{encoder}__{Path(base).name}__{dataset}" + (f"__{name_suffix}" if name_suffix else "")
    out_dir = Path(out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "projector.pt"

    # Resume a crashed/pre-empted run: restore projector + optimizer + step, and rejoin the W&B run.
    resume_ckpt = out_dir / "resume.pt"
    start_step, resume_id = 0, None
    if resume and resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location="cpu")  # load_state_dict places opt state on the param device
        adapter.load_adapter_state_dict(state.get("adapter") or state["projector"])  # legacy: raw projector sd
        opt.load_state_dict(state["opt"])
        start_step, resume_id = state["step"], state.get("wandb_id")
        print(f"resuming from step {start_step}/{max_steps}", flush=True)
    elif init_projector:  # warm-start the projector from a prior best.pt, fresh optimizer + step 0
        sd = torch.load(init_projector, map_location=adapter.llm.device)
        adapter.projector.load_state_dict(sd["projector"] if "projector" in sd else sd)
        print(f"warm-started projector from {init_projector} (fresh optimizer, step 0)", flush=True)

    # Offline when no WANDB_API_KEY (still produces a local trace to sync later); online otherwise.
    run = None
    if use_wandb:
        if not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")
        # W&B notes: the run's own --description (its hypothesis/intent) leads, then an auto param
        # summary for reproducibility. Without --description it's the summary alone.
        summary = (
            f"Stage-1 projector SFT — {encoder} → {projector_kind} projector → frozen "
            f"{Path(base).name}. Corpus: {dataset}{f' ({mixture})' if mixture else ''}; "
            f"lr={lr}, max_steps={max_steps}, effective_batch={grad_accum}. Projector-only "
            f"(base + encoder frozen); vision spliced at <image>."
            + (f" Held-out eval: {eval_dataset}." if eval_dataset else "")
            + (" Multi-turn multimodal QA probe on (qa/metrics/*)." if qa_eval else "")
        )
        notes = f"{description}\n\n{summary}" if description else summary
        run = wandb.init(project="laguna-mm-adapter", name=run_name, notes=notes,
                         id=resume_id, resume="allow" if resume_id else None,
                         config={"base": base, "encoder": encoder, "projector": projector_kind,
                                 "dataset": dataset, "lr": lr, "max_steps": max_steps,
                                 "grad_accum": grad_accum, "n_train": n_train, "seed": seed})
        if start_step == 0:  # sample tables log at step 0 — skip when rejoining a resumed run
            _log_samples(run, train_ds, "train/samples")
            _log_samples(run, val_ds, "val/samples")
            if qa_eval:
                _log_qa_samples(run)  # example QA episodes (the target task) next to qa/metrics/*

    # GPUMonitor samples nvidia-smi in the background; always log a results.jsonl row (even on
    # crash/SIGINT) so the sweep tracker sees CRASH instead of a silently-vanished iter.
    t0 = time.monotonic()
    step, last_loss, last_val, eval_loss, status = (
        start_step, float("nan"), float("nan"), float("nan"), "CRASH")
    best_val, best_qa, since_improve, stop = float("inf"), -1.0, 0, False
    qa_acc, qa_recall = float("nan"), float("nan")
    best_ckpt = out_dir / "best.pt"
    monitor = GPUMonitor()
    try:
        with monitor:
            opt.zero_grad()
            window = None  # GPU-side running sum of micro-losses → mean over the effective batch
            cwin: dict = {}  # per-corpus (loss-sum, count) over the batch, for per-corpus train logging
            while step < max_steps and not stop:
                for images, labels, corpora, questions in loader:
                    loss = _loss(adapter, images, labels, corpora, objective, questions) / grad_accum
                    loss.backward()
                    window = loss.detach() if window is None else window + loss.detach()
                    c = corpora[0]  # micro_batch=1 → one corpus per step
                    if c is not None:
                        s, k = cwin.get(c, (None, 0))
                        cwin[c] = (loss.detach() if s is None else s + loss.detach(), k + 1)
                    if (step + 1) % grad_accum == 0:
                        opt.step()
                        opt.zero_grad()
                        # log the effective-batch mean loss per optimizer step — smoother than a
                        # single micro-example, and one CUDA sync per step rather than every micro-step.
                        last_loss = window.item()
                        if run:
                            log = {"train/loss/total": last_loss}
                            for cc, (s, k) in cwin.items():
                                log[f"train/loss/{cc}"] = (s * grad_accum / k).item()
                            run.log(log, step=step)
                        if (step + 1) % (20 * grad_accum) == 0:
                            print(f"step {step}/{max_steps} loss {last_loss:.4f}", flush=True)
                        window, cwin = None, {}
                    if step % val_every == 0:
                        last_val, val_by_corpus = _val_loss(adapter, val_loader, objective)
                        metrics = {"val/loss/total": last_val}
                        metrics.update({f"val/loss/{cc}": v for cc, v in val_by_corpus.items()})
                        if eval_loader is not None:  # trend the fixed external eval alongside val
                            eval_loss, _ = _val_loss(adapter, eval_loader, objective)
                            metrics["eval/loss/total"] = eval_loss
                        cur_qa = None  # for QA, score read accuracy EVERY val — it's the selection signal
                        if qa_eval and objective == "qa":
                            torch.cuda.empty_cache()  # reclaim training fragmentation before generation
                            metrics.update(dataset_qa_accuracy(adapter, qa_eval_items))
                            cur_qa = metrics["qa/metrics/accuracy"]
                        # SELECT + early-stop on the TASK metric (qa_acc), not val loss: min-CE optimizes
                        # confabulation — val loss is anti-correlated with reading and best-on-val saves
                        # the *worst* reader (W&B: qa peaks early then decays as val keeps falling).
                        improved = (cur_qa > best_qa + min_delta) if cur_qa is not None else (last_val < best_val - min_delta)
                        if improved:
                            since_improve = 0
                            best_val, best_qa = min(best_val, last_val), max(best_qa, cur_qa or -1.0)
                            torch.save(adapter.adapter_state_dict(), best_ckpt)
                        else:
                            since_improve += 1
                        metrics.update({"val/loss/best": best_val, "qa/metrics/best": best_qa})
                        if step % gen_every == 0:  # WER/CER + embed_norm (slow generation) on a coarser cadence
                            torch.cuda.empty_cache()  # reclaim training fragmentation before generation (OOM margin)
                            metrics.update(generation_metrics(adapter, wer_items, "val"))
                            metrics.update(generation_metrics(adapter, ocr_probe, "val/ocr"))  # base-OCR retention
                            # base-preservation gauge — projected tokens in-distribution vs base embeds
                            metrics["val/metrics/embed_norm_ratio"] = adapter.embedding_norm_ratio(wer_images)
                            if read_probe_items:  # real-VQA reading-transfer probe (recon Stage-1) —
                                # observational (read/, no selection); on the slow-gen cadence with WER/CER
                                metrics.update(dataset_qa_accuracy(adapter, read_probe_items, prefix="read"))
                            if run:  # (image, target, prediction) samples — what it reads, over training
                                _log_predictions(run, adapter, wer_items, "val/samples", step)
                        gen_str = ""  # surface the generation-cadence metrics in the text log, not just W&B
                        if "val/metrics/wer" in metrics:
                            gen_str += f"  wer {metrics['val/metrics/wer']:.3f} cer {metrics['val/metrics/cer']:.3f}"
                        if "val/metrics/embed_norm_ratio" in metrics:
                            gen_str += f"  embed_norm {metrics['val/metrics/embed_norm_ratio']:.3f}"
                        if "qa/metrics/accuracy" in metrics:  # full-dataset read acc + per-corpus breakdown
                            gen_str += f"  qa_acc {metrics['qa/metrics/accuracy']:.3f}"
                            gen_str += "".join(f" {k.rsplit('_', 1)[1]}={v:.2f}"
                                               for k, v in metrics.items() if "/acc_" in k)
                        if "read/metrics/accuracy" in metrics:  # inline real-VQA reading-transfer probe
                            gen_str += f"  read_acc {metrics['read/metrics/accuracy']:.3f}"
                        # qa_acc is noisy (small eval set) — it wobbles down between peaks, so a tight
                        # patience early-stops too soon (NaFlex died at step 1119). Be 3x more tolerant
                        # on qa than on the smooth val-loss signal.
                        eff_patience = patience * 3 if objective == "qa" else patience
                        sel = f"best qa {best_qa:.3f}" if objective == "qa" else f"best val {best_val:.4f}"
                        print(f"  val {last_val:.4f} ({sel}, {since_improve}/{eff_patience}){gen_str}", flush=True)
                        if run:
                            run.log(metrics, step=step)
                        if since_improve >= eff_patience:
                            sig = "qa_acc" if objective == "qa" else "val"
                            print(f"early stop: no {sig} improvement in {eff_patience} evals", flush=True)
                            stop = True
                    step += 1
                    if step % val_every == 0:  # crash-recovery checkpoint at the val cadence
                        _save_resume(resume_ckpt, adapter, opt, step, run)
                    if step >= max_steps or stop:
                        break
            # restore the best-val projector — the deliverable + final eval reflect the val minimum,
            # not the (possibly overfit) last step.
            if best_ckpt.exists():
                adapter.load_adapter_state_dict(torch.load(best_ckpt, map_location=adapter.llm.device))
            last_val = best_val
            if eval_loader is not None:  # final ranker value, recomputed on the best checkpoint
                eval_loss, _ = _val_loss(adapter, eval_loader, objective)
                drift = adapter.embedding_norm_ratio(wer_images)
                ocr = generation_metrics(adapter, ocr_probe, "eval/ocr")  # base-OCR retention on held-out probe
                print(f"  eval/{eval_dataset} loss {eval_loss:.4f}  embed_norm_ratio {drift:.3f}"
                      f"  ocr_wer {ocr['eval/ocr/metrics/wer']:.3f}", flush=True)
                if run:
                    run.log({"eval/loss/total": eval_loss, "eval/metrics/embed_norm_ratio": drift, **ocr}, step=step)
                    _log_predictions(run, adapter, ocr_probe, "eval/samples", step)
            if qa_eval:  # single-turn read accuracy (per corpus) + multi-turn cross-turn recall
                qa = dataset_qa_accuracy(adapter, qa_eval_items)  # the target metric (incl. VQA/synthetic)
                qa_acc = qa["qa/metrics/accuracy"]
                per_c = "  ".join(f"{k.rsplit('_', 1)[1]} {v:.2f}" for k, v in qa.items() if "/acc_" in k)
                # multi-turn: read A, read B, recall A — qa_mt/recall = conversation memory (cross-turn)
                mt = evaluate_multiturn_qa(adapter, n=12, source="mixture", prefix="qa_mt")
                qa.update(mt)
                qa_recall = mt["qa_mt/metrics/recall"]
                print(f"  full-dataset QA: acc {qa_acc:.3f}  [{per_c}]", flush=True)
                print(f"  multi-turn: acc {mt['qa_mt/metrics/accuracy']:.3f}  recall {qa_recall:.3f} "
                      "(cross-turn memory)", flush=True)
                if run:
                    run.log(qa, step=step)
        torch.save(adapter.adapter_state_dict(), ckpt)  # = best-val weights
        print(f"saved adapter -> {ckpt} (best val {best_val:.4f})", flush=True)
        status = "KEEP"
        resume_ckpt.unlink(missing_ok=True)  # completed — don't let a fresh run resume it
    finally:
        gpu = monitor.summary()
        print(monitor.format_summary(), flush=True)
        if run:
            run.summary.update({"final_loss": last_loss, "val_loss": last_val, "eval_loss": eval_loss,
                                "qa_accuracy": qa_acc, "qa_recall": qa_recall, "status": status,
                                "gpu_peak_mem_gb": gpu.peak_mem_gb, "gpu_mean_util_pct": gpu.mean_util_pct})
            run.finish()
        log_experiment(
            tag="mm_adapter", config_name=run_name,
            score=last_loss, steps=step, status=status,
            description=f"visual projector ({projector_kind}) on {base} [{dataset}]",
            wandb_url=run.url if run else "",
            runtime_min=(time.monotonic() - t0) / 60,
            extra={"final_loss": last_loss, "val_loss": last_val, "eval_loss": eval_loss,
                   "qa_accuracy": qa_acc, "qa_recall": qa_recall,
                   "eval_dataset": eval_dataset, "backbone": base,
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
    dataset: str = typer.Option("synthetic", help=" | ".join(CHOICES)),
    wandb_tracking: bool = typer.Option(True, "--wandb/--no-wandb", help="log to Weights & Biases"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="resume from resume.pt if present"),
    mixture: str = typer.Option("", help="override mix weights, e.g. 'websight=0.6,webcode2m=0.4'"),
    name_suffix: str = typer.Option("", help="appended to the run name (isolates sweep variants)"),
    eval_dataset: str = typer.Option("", help="fixed held-out eval corpus, e.g. design2code (logs eval/loss)"),
    patience: int = typer.Option(3, help="early-stop after this many evals without val improvement"),
    min_delta: float = typer.Option(1e-3, help="min val/loss decrease to count as an improvement"),
    qa_eval: bool = typer.Option(True, "--qa-eval/--no-qa-eval", help="multi-turn multimodal QA probe (slow)"),
    description: str = typer.Option("", help="this run's hypothesis/intent — leads the W&B run notes"),
    init_projector: str = typer.Option("", help="warm-start the projector from a prior best.pt (fresh optimizer)"),
    objective: str = typer.Option("recon", help="recon (transcribe) | qa (QA-SFT — forces vision use)"),
    unfreeze: str = typer.Option("", help="'' = projector only | lora = + attention LoRA on the frozen LLM"),
    anchor: bool = typer.Option(True, "--anchor/--no-anchor", help="soft-scalar norm match on vision tokens"),
    lr: float = typer.Option(None, help="override the config learning rate"),
    vqa: str = typer.Option("default", help="VQA reading sets for QA-SFT: 'default'=all, comma-list, or '' = none"),
    norm_penalty: float = typer.Option(0.0, help="soft cap on projected-token scale (--no-anchor ballooning)"),
) -> None:
    train(config, encoder, base, steps, n_train, pool, projector, out, seed, dataset, wandb_tracking,
          resume, mixture, name_suffix, eval_dataset, patience, min_delta, qa_eval, description, init_projector,
          objective, unfreeze, anchor, lr, vqa, norm_penalty)


if __name__ == "__main__":
    app()
