"""Tool-trace SFT warm-start — teach the adapter to drive the encoder+decoder+tool loop.

The SFT adapter answers single-shot (bake-off encoder_tool 0.36) but emits `</assistant>` under a tool
prompt — it lacks the tool-call *format* + multi-turn structure, not the answering capability. Warm-start it
on golden traces (the EXACT prompt/observation VisionToolEnv uses, via the shared helpers), so it learns to
emit `poolside` calls across turns; then RLVR shapes per-item trust. See
docs/specs/2026-06-11-tool-trace-sft-warmstart-design.md.

Per-item-trust split (self-supervised): a trace teaches ocr-then-answer when the transcript actually carries
the gold (`_match`), and direct-answer (no ocr) when it doesn't — so the model learns *when* to read text vs
answer from the image, not "always call ocr."

    uv run python -m laguna_rlvr.visual.vision_tool_sft train <ckpt> --n 40 --epochs 3
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import typer

from laguna_rlvr.scaffold import format_call
from laguna_rlvr.visual.model import IMAGE_TOKEN
from laguna_rlvr.visual.multiturn_qa import _match
from laguna_rlvr.visual.ocr_backend_eval import _glyph_corpora
from laguna_rlvr.visual.tool_eval import load_items
from laguna_rlvr.visual.vision_tool_eval import (_glyph_transcripts, _load_adapter, episode_prompt,
                                                 ocr_observation)

Segment = tuple[str, bool]  # (text, supervised) — supervised spans are the tool calls the model must emit


def synth_trace(image_id: str, question: str, gold: str, transcript: str, fmt: str,
                use_ocr: bool) -> list[Segment]:
    """One golden trace as (text, supervised) segments. The turn-0 prompt (with <image>) and the ocr
    observation are the SHARED VisionToolEnv helpers, so warm-start trains on the exact eval prompt.
    Only the tool-call spans are supervised; the prompt + observation are context."""
    prompt = (episode_prompt(image_id, question, fmt), False)
    answer = (format_call("answer", "value", gold, fmt), True)
    if not use_ocr:
        return [prompt, answer]
    return [prompt,
            (format_call("ocr", "image_id", image_id, fmt), True),
            ("\n" + ocr_observation(image_id, transcript) + "\n", False),
            answer]


def synth_traces(n: int, *, fmt: str) -> list[tuple[object, list[Segment]]]:
    """(image, segments) per glyph item. use_ocr is decided by whether the transcript actually carries the
    gold — the self-supervised per-item-trust signal (read text when it helps, answer from vision when not)."""
    items = load_items(_glyph_corpora(), n)
    transcripts = _glyph_transcripts(n)
    seen: dict[str, int] = {}
    traces = []
    for corpus, img, q, gold in items:
        i = seen.get(corpus, 0)
        seen[corpus] = i + 1
        transcript = transcripts.get((corpus, i), "")
        use_ocr = _match(str(gold), transcript)  # the transcript carries the answer -> teach ocr-then-answer
        traces.append((img, synth_trace(f"{corpus}.png", q, str(gold), transcript, fmt, use_ocr)))
    return traces


def _trace_seq(adapter, image, segments: list[Segment], emb) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input_embeds (L, D), labels (L,)) for one trace: vision spliced in the prompt, labels −100
    everywhere except the supervised tool-call spans. Grad flows through the projector (vis) + LoRA.
    `emb` is the (frozen) input-embedding layer, hoisted by the caller — it's constant across traces."""
    vis = adapter._project([image])[0:1]  # (1, Nv, D) — projector has grad; frozen encoder doesn't
    dev = adapter.llm.device
    parts, labels = [], []
    for text, supervised in segments:
        if IMAGE_TOKEN in text:
            e = adapter._embed_multi(text, [vis])                       # (1, T', D), <image> -> Nv tokens
            lab = torch.full((e.shape[1],), -100, dtype=torch.long, device=dev)
        else:
            ids = adapter.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            e = emb(ids)                                                # (1, T, D)
            lab = ids[0].clone() if supervised else torch.full((ids.shape[1],), -100, dtype=torch.long, device=dev)
        parts.append(e[0])
        labels.append(lab)
    return torch.cat(parts, dim=0), torch.cat(labels, dim=0)


def run_train(ckpt: str, n: int, *, fmt: str, base: str, epochs: int, lr: float, micro: int,
              seed: int, out: Path) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    adapter = _load_adapter(ckpt, base)
    adapter.train()
    traces = synth_traces(n, fmt=fmt)
    n_ocr = sum(1 for _, segs in traces if len(segs) == 4)
    print(f"[tool-sft] {len(traces)} traces — {n_ocr} ocr-then-answer, {len(traces) - n_ocr} direct", flush=True)
    emb = adapter.llm.get_input_embeddings()  # frozen embedding layer — constant across all traces
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr)  # single group: LoRA only (base frozen)
    order = list(range(len(traces)))
    for ep in range(epochs):
        random.shuffle(order)
        tot, nb = 0.0, 0
        for s0 in range(0, len(order), micro):
            seqs, labs = [], []
            for i in order[s0:s0 + micro]:
                seq, lab = _trace_seq(adapter, traces[i][0], traces[i][1], emb)
                seqs.append(seq)
                labs.append(lab)
            loss = adapter._batched_lm_loss(seqs, labs, adapter.llm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            tot += loss.item()
            nb += 1
        print(f"[tool-sft] epoch {ep + 1}/{epochs} mean-loss {tot / max(nb, 1):.4f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.adapter_state_dict(), out)
    print(f"[tool-sft] saved warm-started adapter -> {out}", flush=True)


app = typer.Typer(add_completion=False, help=__doc__)


@app.callback()
def _main() -> None:
    """Group callback so `train` stays an explicit subcommand."""


@app.command()
def train(
    ckpt: str = typer.Argument(..., help="SFT adapter checkpoint to warm-start from"),
    n: int = typer.Option(40, help="items per glyph corpus"),
    fmt: str = typer.Option("poolside", help="tool-call scaffold to teach"),
    base: str = typer.Option("poolside/Laguna-XS.2"),
    epochs: int = typer.Option(3),
    lr: float = typer.Option(1e-4),
    micro: int = typer.Option(2, help="traces per grad step"),
    seed: int = typer.Option(0),
    out: str = typer.Option("results/visual/glm_ocr__tool_sft/best.pt", help="warm-started checkpoint path"),
) -> None:
    """Warm-start the adapter on golden tool-call traces so it can drive VisionToolEnv."""
    run_train(ckpt, n, fmt=fmt, base=base, epochs=epochs, lr=lr, micro=micro, seed=seed, out=Path(out))


if __name__ == "__main__":
    app()
