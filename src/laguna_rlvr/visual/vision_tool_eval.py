"""Vision tool-env: the agentic encoder+decoder+tool loop, run on the LOCAL adapter.

The text envs (ocr_tool, …) run through `prime eval` against a remote text API, so they can only reach
vision *through a text tool* (decoder+tool). This env closes the other channel — **encoder+decoder+tool** —
by driving the loop on the local `VisualAdapter`: the image is spliced into the decoder (the encoder
channel, always on), AND the model can still call `ocr` for the glyph transcript, then `answer`. It is the
agentic form of the `encoder_tool` config that won the architecture bake-off ([[bakeoff]]), now as a
probe-gated member of the portfolio.

Vision can't ride the remote `prime eval` path (its API is text-only), so the policy IS the adapter and the
loop runs in-process: greedy-generate (vision spliced) -> parse a text-scaffold tool call -> serve the
observation -> repeat. It writes `results/probe/vision_tool__<slug>.jsonl` in the same {success, reward}
shape `report.py` ranks, so it folds into the same probe -> rank -> train pipeline as the text envs.

    uv run python -m laguna_rlvr.visual.vision_tool_eval probe <ckpt> --n 40 --fmt poolside
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
import typer

from laguna_rlvr.rewards import RolloutState, binary, shaped
from laguna_rlvr.scaffold import FORMATS, Tool, parse_call, render_instructions
from laguna_rlvr.visual.model import IMAGE_TOKEN, VisualAdapter
from laguna_rlvr.visual.multiturn_qa import _match
from laguna_rlvr.visual.ocr_backend_eval import _csv, _ensure, _glyph_corpora
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.tool_eval import DEFAULT_VQA, load_items

_NO_OCR = "(no OCR text extracted from this image — read it from the picture)"

# Same two-tool pool as ocr_tool, so the only axis vs that env is the added vision channel (the image
# is spliced) — the model can read glyphs off the pixels OR call ocr for the transcript, then answer.
_TOOLS = [Tool("ocr", "image_id", "returns the text an OCR encoder extracts from that image"),
          Tool("answer", "value", "submit your final answer — just the value")]


def _interpret(reply: str, fmt: str, gold: str, transcript: str) -> tuple[str, object]:
    """Pure loop step: map a model reply to ('done', solved_bool) if it answered, else ('continue',
    observation_text) — an ocr result or a re-prompt. Adapter-free, so the loop control is unit-tested
    without touching the GPU; run_episode supplies the generation around it."""
    call = parse_call(reply, fmt, _TOOLS)
    if call and call[0] == "answer":
        return "done", _match(gold, call[1])
    if call and call[0] == "ocr":
        return "continue", f"[ocr of {call[1]}]\n{transcript}\n\nNow answer, or call a tool."
    return "continue", "No valid tool call found.\n" + render_instructions(_TOOLS, fmt)


@torch.no_grad()
def run_episode(adapter: VisualAdapter, image, image_id: str, question: str, transcript: str, gold: str,
                *, fmt: str = "poolside", max_turns: int = 4, max_new_tokens: int = 24) -> tuple[bool, int]:
    """One encoder+decoder+tool episode. Turn 0 splices the image (the encoder channel) alongside the
    question + tool instructions; each later turn appends the tool observation. Mirrors `VisualAdapter.chat`'s
    context accumulation, but the next turn is decided by parsing the model's reply (`_interpret`), not given.
    Caller disables gradient checkpointing once around the whole run (see run_probe) — generation only."""
    instr = render_instructions(_TOOLS, fmt)
    prompt = (f"You can SEE the image (id: {image_id}) below. You may also call ocr on that id to read "
              f"its text, then answer.\n{IMAGE_TOKEN}\n\nTask: {question}\n\n{instr}")
    vis = adapter._project([image])[0:1]  # (1, Nv, D) — the encoder channel
    ctx = adapter._embed_multi(prompt, [vis])
    emb = adapter.llm.get_input_embeddings()
    for turn in range(1, max_turns + 1):
        gen = adapter.llm.generate(inputs_embeds=ctx, max_new_tokens=max_new_tokens, do_sample=False)
        reply = adapter.tok.decode(gen[0], skip_special_tokens=True).strip()
        ctx = torch.cat([ctx, emb(gen)], dim=1)
        kind, payload = _interpret(reply, fmt, gold, transcript)
        if kind == "done":
            return bool(payload), turn
        ctx = torch.cat([ctx, adapter._embed_multi(f"\n{payload}\n", [])], dim=1)
    return False, max_turns


def _load_adapter(ckpt: str, base: str) -> VisualAdapter:
    adapter = VisualAdapter(load_encoder("glm_ocr", pool=2), base, projector_kind="resampler",
                            use_anchor=True, unfreeze="lora", lora_rank=128, n_queries=256)
    adapter.load_adapter_state_dict(torch.load(ckpt, map_location=adapter.llm.device))
    adapter.eval()
    return adapter


def _glyph_transcripts(n: int) -> dict[tuple[str, int], str]:
    """Qwen3-VL glyph transcripts keyed by (corpus, within-corpus index). A full-matrix run looks up the
    OCR text for glyph corpora and falls back to a no-text marker on non-glyph (chart/general) images —
    where there's nothing to transcribe and the encoder channel carries the signal instead."""
    by_key: dict[tuple[str, int], str] = {}
    seen: dict[str, int] = defaultdict(int)
    for r in _ensure("qwen3_vl", "glyph", n, "cuda"):  # {corpus, gold, transcript}
        c = r["corpus"]
        by_key[(c, seen[c])] = r["transcript"]
        seen[c] += 1
    return by_key


def run_probe(ckpt: str, n: int, *, vqa_names: list[str], fmt: str, max_turns: int, base: str,
              eff_w: float, slug: str, out: Path) -> dict[str, float]:
    """Drive the vision tool-loop over `vqa_names` and write portfolio records. The image is always spliced
    (encoder channel); the ocr() tool serves the bake-off's cached Qwen3-VL transcript on glyph corpora,
    a no-text marker elsewhere — so the full 12-task matrix exercises encoder+tool where text helps and
    encoder-alone where it doesn't (charts/general)."""
    items = load_items(vqa_names, n)
    transcripts = _glyph_transcripts(n)
    adapter = _load_adapter(ckpt, base)
    adapter.llm.gradient_checkpointing_disable()  # generation only — toggle once around the whole run
    out.parent.mkdir(parents=True, exist_ok=True)
    hits: dict[str, list[int]] = {}
    seen: dict[str, int] = defaultdict(int)
    try:
        with out.open("w") as f:
            for corpus, img, q, gold in items:
                tr = transcripts.get((corpus, seen[corpus]), _NO_OCR)
                seen[corpus] += 1
                solved, turns = run_episode(adapter, img, f"{corpus}.png", q, tr, str(gold),
                                            fmt=fmt, max_turns=max_turns)
                rs = RolloutState(tests_passed=int(solved), tests_total=1, turns=turns,
                                  max_turns=max_turns, succeeded=solved)
                f.write(json.dumps({"env": "vision_tool", "model": slug, "corpus": corpus,
                                    "success": bool(solved), "reward": shaped(rs, eff_w),
                                    "_success": binary(rs)}) + "\n")
                cell = hits.setdefault(corpus, [0, 0])
                cell[0] += int(solved)
                cell[1] += 1
    finally:
        adapter.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return {c: h / nn for c, (h, nn) in hits.items()}


app = typer.Typer(add_completion=False, help=__doc__)


@app.callback()
def _main() -> None:
    """Group callback so `probe` stays an explicit subcommand (Typer collapses single-command apps)."""


@app.command()
def probe(
    ckpt: str = typer.Argument(..., help="trained glm_ocr adapter checkpoint (encoder_tool's decoder)"),
    n: int = typer.Option(40, help="items per corpus"),
    vqa: str = typer.Option("glyph", help="'glyph' (7 OCR-answerable), 'all' (the 12-task matrix), or a comma list"),
    fmt: str = typer.Option("poolside", help=f"text tool-call scaffold the local adapter emits: {list(FORMATS)}"),
    max_turns: int = typer.Option(4),
    base: str = typer.Option("poolside/Laguna-XS.2"),
    eff_w: float = typer.Option(0.1, help="efficiency weight in the shaped reward"),
    slug: str = typer.Option("laguna", help="model slug for the probe record filename"),
    out: str = typer.Option("", help="record path (default results/probe/vision_tool__<slug>.jsonl)"),
) -> None:
    """Probe the encoder+decoder+tool loop on the local adapter; writes a portfolio-ranked record."""
    if fmt not in FORMATS:
        raise typer.BadParameter(f"local adapter emits text, so fmt must be one of {list(FORMATS)} (not native)")
    vqa_names = _glyph_corpora() if vqa == "glyph" else (_csv(DEFAULT_VQA) if vqa == "all" else _csv(vqa))
    out_path = Path(out) if out else Path(f"results/probe/vision_tool__{slug}.jsonl")
    per_corpus = run_probe(ckpt, n, vqa_names=vqa_names, fmt=fmt, max_turns=max_turns, base=base,
                           eff_w=eff_w, slug=slug, out=out_path)
    overall = sum(per_corpus.values()) / max(len(per_corpus), 1)
    print(f"\n[vision_tool] overall {overall:.2f} | " +
          " ".join(f"{c} {v:.2f}" for c, v in sorted(per_corpus.items())), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    app()
