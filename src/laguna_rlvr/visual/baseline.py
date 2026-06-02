"""Stage-0 baselines: what does text-only Laguna do on the visual tasks *before* any adapter?

Two no-adapter baselines over the frozen base LLM — `blind` (task prompt only) and tool-mediated
(`GLM-OCR → text → Laguna`) — give the adapter's eventual numbers a floor and a bar to beat. See
docs/specs/2026-05-30-stage-0-baseline-design.md.

Staged GPU: GLM-OCR transcribes every image first and is freed before the base LLM loads, so the two
multi-GB models never co-reside on the A100.
"""
from __future__ import annotations

import gc
import os

import torch
import typer
from transformers import AutoModelForImageTextToText, AutoProcessor

from laguna_rlvr.visual.corpora import CORPUS_KIND, TASK_PROMPT, load_text_image
from laguna_rlvr.visual.encoders import _REPOS  # canonical model-id registry (avoid a 2nd source of truth)
from laguna_rlvr.visual.metrics import _CODE_GEN_TOKENS, _OCR_GEN_TOKENS, score_predictions  # gen budgets: one source
from laguna_rlvr.visual.model import load_causal_lm

_OCR_PROMPT = "Transcribe all text visible in this image, preserving order."
_QA_GEN_TOKENS = 24      # multi-turn QA replies are short ("the title is …")

app = typer.Typer(add_completion=False)


@torch.no_grad()
def glm_ocr_transcribe(items: list, device: str = "cuda", max_new_tokens: int = 48,
                       model=None, proc=None) -> list[str]:
    """GLM-OCR reads each item's image (image -> OCR text). One transcript per item, in order.

    GLM-OCR is an image-text-to-text model: it's prompted with the image *plus* an instruction (via
    the chat template), so the full `AutoProcessor` is required, not a bare image processor. Only the
    generated continuation is decoded (the echoed prompt tokens are sliced off). Loads the model once
    when `model`/`proc` are omitted, so one call with the full item list is the cheap path (the
    staged-GPU harness frees it before loading Laguna). `items` are (image, ...) tuples; only `it[0]`.
    """
    if model is None:
        repo = _REPOS["glm_ocr"]
        model = AutoModelForImageTextToText.from_pretrained(repo, device_map=device).eval()
        proc = AutoProcessor.from_pretrained(repo)
    out = []
    for it in items:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": it[0]}, {"type": "text", "text": _OCR_PROMPT}]}]
        batch = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt").to(model.device)
        gen = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        new = gen[:, batch["input_ids"].shape[1]:]  # drop the echoed prompt; keep the transcript
        out.append(proc.batch_decode(new, skip_special_tokens=True)[0])
    return out


def assemble_prompt(task: str, transcript: str | None) -> str:
    """Build one base-LLM prompt. `blind` (transcript None) is the task alone — the text-only floor;
    tool-mediated prepends the OCR transcript as the model's only window onto the image.
    """
    if transcript is None:
        return task
    return f"{transcript}\n\n{task}"


@torch.no_grad()
def text_generate(llm, tok, prompts: list[str], max_new_tokens: int = _CODE_GEN_TOKENS) -> list[str]:
    """Greedy text-only completion of each prompt (no vision splice). Returns only the continuation —
    the echoed prompt tokens are stripped so the prediction is the model's answer, not the question.
    """
    out = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(llm.device)
        gen = llm.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
        out.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    return out


@torch.no_grad()
def text_chat(llm, tok, turn_texts: list[str], max_new_tokens: int = _QA_GEN_TOKENS) -> list[str]:
    """Multi-turn text-only chat: one reply per turn, each conditioned on all prior turns + replies.
    The text-only mirror of `VisualAdapter.chat` (ids instead of spliced embeds) — the blind/OCR
    engine for the multi-turn QA axis.
    """
    ctx, replies = None, []
    for text in turn_texts:
        ids = tok(text, return_tensors="pt").input_ids.to(llm.device)
        ctx = ids if ctx is None else torch.cat([ctx, ids], dim=1)
        gen = llm.generate(input_ids=ctx, max_new_tokens=max_new_tokens, do_sample=False)
        reply_ids = gen[:, ctx.shape[1]:]
        replies.append(tok.decode(reply_ids[0], skip_special_tokens=True))
        ctx = torch.cat([ctx, reply_ids], dim=1)
    return replies


def run_panels(dataset: str, baselines: list[str], n_eval: int, base_llm: str,
               qa: bool, device: str = "cuda") -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Both Stage-0 axes in ONE staged-GPU pass: GLM-OCR transcribes every image (Axis A corpus items
    + Axis B episode images) and is freed, then a SINGLE base LLM scores both axes. Loading a base per
    axis put two 33B models on one 80GB A100 and OOM'd — one shared base is the fix (and halves load).
    Returns (axis_a single-turn, axis_b multi-turn QA — empty when `qa` is False).
    """
    # local import: the QA library imports this module for its engine primitives; top-level would cycle.
    from laguna_rlvr.visual.multiturn_qa import (
        blind_runner, image_fetcher, load_or_build_episodes, ocr_runner, run_qa, transcribe_episodes)

    items = list(load_text_image(dataset, n_eval))
    kind = CORPUS_KIND.get(dataset)
    task = TASK_PROMPT.get(kind, TASK_PROMPT[None])
    a_max = _OCR_GEN_TOKENS if kind is None else _CODE_GEN_TOKENS
    episodes = load_or_build_episodes("mixture", n_eval, 0) if qa else []
    want_ocr = "ocr" in baselines

    a_transcripts = b_transcripts = None
    if want_ocr:  # stage 1: ONE GLM-OCR session over every image (both axes), then freed
        a_transcripts = glm_ocr_transcribe(items, device=device)
        if qa:
            b_transcripts = transcribe_episodes(episodes, image_fetcher(episodes), device=device)
        gc.collect()
        torch.cuda.empty_cache()

    llm, tok = load_causal_lm(base_llm, device, torch.bfloat16)  # stage 2: ONE base for both axes

    refs, kinds = [it[1] for it in items], [kind] * len(items)
    axis_a: dict[str, dict[str, float]] = {}
    for name in baselines:
        per_item = a_transcripts if name == "ocr" else [None] * len(items)  # blind = no transcript
        preds = text_generate(llm, tok, [assemble_prompt(task, t) for t in per_item], max_new_tokens=a_max)
        axis_a[name] = score_predictions(preds, refs, kinds, prefix=f"baseline/{name}")

    axis_b: dict[str, dict[str, float]] = {}
    if qa:
        for name in baselines:
            if name == "blind":
                run = blind_runner(llm, tok)
            elif name == "ocr":
                run = ocr_runner(llm, tok, b_transcripts)
            else:
                continue
            axis_b[name] = run_qa(run, episodes, prefix=f"qa/{name}")
    return axis_a, axis_b


def _print_results(dataset: str, results: dict[str, dict[str, float]]) -> None:
    for name, metrics in results.items():
        body = " ".join(f"{k.rsplit('/', 1)[-1]}={v:.3f}" for k, v in sorted(metrics.items()))
        print(f"RESULT dataset={dataset} baseline={name} {body}", flush=True)


def _log_wandb(run, panels: dict[str, dict[str, dict[str, float]]]) -> None:
    """Log each axis as a baseline×metric comparison Table (rows = baselines, cols = short metric
    names) — so blind vs OCR read off one view, not scattered nested scalars — plus the flat scalars
    so a metric trends across runs (chartmimic → design2code → the adapter)."""
    import wandb
    flat: dict[str, float] = {}
    for axis, panel in panels.items():
        if not panel:  # warn rather than silently drop — an empty QA panel means no episodes built
            print(f"WARN: '{axis}' panel empty — nothing logged for it (built no episodes/results?)",
                  flush=True)
            continue
        cols = sorted({k.rsplit("/", 1)[-1] for m in panel.values() for k in m})
        table = wandb.Table(columns=["baseline", *cols])
        for name, metrics in panel.items():
            short = {k.rsplit("/", 1)[-1]: v for k, v in metrics.items()}
            table.add_data(name, *(short.get(c) for c in cols))
            flat.update(metrics)
        run.log({f"{axis}/summary": table})
    run.log(flat)


@app.command()
def baseline(
    dataset: str = typer.Option("design2code", help="corpus to evaluate (Axis A)"),
    baselines: str = typer.Option("blind,ocr", help="comma list: blind, ocr"),
    base: str = typer.Option(..., help="base LLM checkpoint (frozen, no adapter)"),
    n_eval: int = typer.Option(64, help="held-out items to score"),
    device: str = typer.Option("cuda"),
    qa: bool = typer.Option(True, "--qa/--no-qa", help="also run Axis B (multi-turn multimodal QA)"),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb", help="log to Weights & Biases"),
) -> None:
    """Stage-0 baseline panel: frozen base LLM, blind vs OCR-mediated — single-turn (Axis A) + QA (B).
    QA + W&B logging are on by default (--no-qa / --no-wandb to skip)."""
    names = [b.strip() for b in baselines.split(",") if b.strip()]
    axis_a, axis_b = run_panels(dataset, names, n_eval, base, qa, device)
    _print_results(dataset, axis_a)
    if axis_b:
        _print_results("multiturn-qa", axis_b)
    if use_wandb:
        import wandb
        if not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")
        run = wandb.init(project="laguna-mm-adapter", name=f"baseline-{dataset}",
                         config={"base": base, "dataset": dataset, "baselines": names, "n_eval": n_eval})
        _log_wandb(run, {dataset: axis_a, "multiturn-qa": axis_b})
        run.finish()


if __name__ == "__main__":
    app()
