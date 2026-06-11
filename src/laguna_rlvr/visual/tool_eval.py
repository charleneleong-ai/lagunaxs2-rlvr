"""Architecture bake-off on the VQA suite: encoder vs decoder+tool vs encoder+decoder+tool.

Three configs scored over IDENTICAL items + IDENTICAL GLM-OCR transcripts, all on the SAME local
Laguna-XS.2 decoder (decoder-controlled), with the same per-corpus `_match` the adapter eval uses:

  encoder       adapter sees the image, no tool             -> semantic tasks, ~0 on dense glyph
  tool          decoder reads the OCR transcript, no image  -> glyph tasks, ~0 on semantic
  encoder_tool  adapter sees image AND reads transcript      -> the full-breadth config (should get both)

The configs differ ONLY in the per-item prompt (and whether the image is spliced):
  encoder       f"{IMAGE}\n{q}\nAnswer:"                 vision splice
  tool          f"{transcript}\n{q}\nAnswer:"            text-only (no splice)
  encoder_tool  f"{transcript}\n{IMAGE}\n{q}\nAnswer:"    vision splice + transcript

Memory: XS.2 is 63GB and the box has 94GB RAM, so GLM-OCR and the decoder must not co-reside. The
`transcribe` step runs as its OWN process (GLM-OCR memory is reclaimed on exit) and caches transcripts
to disk; the bake-off then loads the decoder ONCE and runs every config through it. All three configs
share the exact same decoder weights (the trained adapter's) — the cleanest decoder-controlled compare;
`tool` simply skips the vision splice. (A pristine-base `tool` variant is a cheap follow-up.)

    uv run python -m laguna_rlvr.visual.tool_eval transcribe --n 40    # GLM-OCR, its own process
    uv run python -m laguna_rlvr.visual.tool_eval bakeoff <ckpt> --n 40
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import typer

from laguna_rlvr.visual.baseline import glm_ocr_transcribe
from laguna_rlvr.visual.corpora import load_vqa, read_question
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import IMAGE_TOKEN, VisualAdapter
from laguna_rlvr.visual.multiturn_qa import _match

# the 12-task matrix from docs/experiments/glm-ocr-encoder/results.md, in that row order
DEFAULT_VQA = ("vqav2,visual7w,figureqa,plotqa,dvqa,textvqa,chartqa,chart2text,"
               "docvqa,ocrvqa,infographic_vqa,visualmrc")
# the glyph corpora the OCR tool can actually answer (the transcript carries the answer) — semantic
# corpora (vqav2/visual7w) are excluded: OCR returns nothing useful, so the loop can only fail them.
GLYPH_VQA = "textvqa,chartqa,dvqa,docvqa,ocrvqa,infographic_vqa,visualmrc"
CONFIGS = ("encoder", "tool", "encoder_tool")
_ANSWER = "\nAnswer:"


def _transcript_path(vqa_names: list[str], n: int) -> Path:
    """Cache transcripts per (corpora, n) so different runs (smoke vs full) never read a stale file."""
    tag = "_".join(vqa_names)[:60]
    return Path(f"results/tool_eval/transcripts__{tag}__n{n}.jsonl")


def assemble_prompt(config: str, transcript: str, question: str) -> str:
    """The `\\nAnswer:` cue matches the adapter eval's gencheck prompt; the per-config forms are the
    prompt table in the module docstring."""
    q = f"{question}{_ANSWER}"
    if config == "encoder":
        return f"{IMAGE_TOKEN}\n{q}"
    if config == "tool":
        return f"{transcript}\n{q}"
    if config == "encoder_tool":
        return f"{transcript}\n{IMAGE_TOKEN}\n{q}"
    raise ValueError(f"unknown config {config!r}; choices: {CONFIGS}")


def accuracy_matrix(hits: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, float]]:
    """hits[config][corpus] = [n_hit, n_total] -> per-corpus accuracy + an 'overall' micro-average."""
    out: dict[str, dict[str, float]] = {}
    for config, per_corpus in hits.items():
        acc = {c: h / max(n, 1) for c, (h, n) in per_corpus.items()}
        tot_h, tot_n = (sum(v) for v in zip(*per_corpus.values())) if per_corpus else (0, 0)
        out[config] = {"overall": tot_h / max(tot_n, 1), **acc}
    return out


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], (prev + 1 if x == y else max(dp[j], dp[j - 1]))
    return dp[len(b)]


def rouge_l_f1(gold: str, pred: str) -> float:
    """LCS-based ROUGE-L F1 (lowercased token overlap). For free-form caption tasks where exact/substring
    `_match` scores a semantically-right-but-differently-worded caption 0 — a metric artifact, not a gap."""
    g, p = gold.lower().split(), pred.lower().split()
    lcs = _lcs_len(g, p)
    if not lcs:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def load_items(vqa_names: list[str], n: int) -> list[tuple]:
    """(corpus, image, question, gold) for the first n items of each VQA corpus — deterministic order,
    so the transcribe pass and the bake-off pass align item-for-item by position."""
    items = []
    for ds, name in load_vqa(vqa_names, n):
        for i in range(min(n, len(ds))):
            img, q, gold = ds[i]
            items.append((name, img, q or read_question(None), gold))
    return items


@torch.no_grad()
def _adapter_answer(adapter: VisualAdapter, config: str, transcript: str, img, question: str,
                    max_new_tokens: int) -> str:
    prompt = assemble_prompt(config, transcript, question)
    if config == "tool":  # text-only: no vision splice, just the transcript + question
        ids = adapter.tok(prompt, return_tensors="pt").input_ids.to(adapter.llm.device)
        g = adapter.llm.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.3)
        return adapter.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
    inp = adapter._embed_with_vision(prompt, adapter._project([img])[0:1])
    g = adapter.llm.generate(inputs_embeds=inp, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=1.3)
    return adapter.tok.decode(g[0], skip_special_tokens=True).split("\n")[0].strip()


def _ensure_transcripts(vqa_names: list[str], n: int, path: Path) -> list[str]:
    """Transcribe in a CHILD process (GLM-OCR's 20GB is reclaimed on exit, so it never co-resides with
    the 63GB decoder), caching to disk. Returns transcripts aligned to load_items() order."""
    if not path.exists():
        cmd = [sys.executable, "-u", "-m", "laguna_rlvr.visual.tool_eval", "transcribe",
               "--vqa", ",".join(vqa_names), "--n", str(n), "--out", str(path)]
        subprocess.run(cmd, check=True)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r["transcript"] for r in rows]


def run_bakeoff(ckpt: str, vqa_names: list[str], n: int, configs: list[str], base: str,
                max_new_tokens: int = 24) -> dict[str, dict[str, float]]:
    items = load_items(vqa_names, n)
    print(f"[bakeoff] {len(items)} items over {len(vqa_names)} corpora; configs={configs}", flush=True)

    need_ocr = any(c in ("tool", "encoder_tool") for c in configs)
    transcripts = (_ensure_transcripts(vqa_names, n, _transcript_path(vqa_names, n))
                   if need_ocr else [""] * len(items))
    assert len(transcripts) == len(items), f"{len(transcripts)} transcripts vs {len(items)} items"

    adapter = VisualAdapter(  # one 63GB load; every config runs through this decoder
        load_encoder("glm_ocr", pool=2), base, projector_kind="resampler",
        use_anchor=True, unfreeze="lora", lora_rank=128, n_queries=256)
    adapter.load_adapter_state_dict(torch.load(ckpt, map_location=adapter.llm.device))
    adapter.eval()

    hits: dict[str, dict[str, list[int]]] = {c: defaultdict(lambda: [0, 0]) for c in configs}
    preds_path = Path("results/tool_eval/preds.jsonl")
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    with preds_path.open("w") as pf:  # auditable: every (config, corpus, gold, pred, hit)
        for config in configs:
            for t, (corpus, img, q, gold) in zip(transcripts, items):
                pred = _adapter_answer(adapter, config, t, img, q, max_new_tokens)
                hit = int(_match(gold, pred))
                cell = hits[config][corpus]
                cell[0] += hit
                cell[1] += 1
                pf.write(json.dumps({"config": config, "corpus": corpus, "gold": gold,
                                     "pred": pred, "hit": hit}) + "\n")
            print(f"[bakeoff] {config} done", flush=True)
    return accuracy_matrix(hits)


def _render(matrix: dict[str, dict[str, float]], vqa_names: list[str], configs: list[str]) -> str:
    head = f"| {'task':<16} | " + " | ".join(f"{c:<13}" for c in configs) + " |"
    sep = "|" + "-" * 18 + "|" + "|".join("-" * 15 for _ in configs) + "|"
    rows = [head, sep]
    for task in ["overall", *vqa_names]:
        cells = " | ".join(f"{matrix[c].get(task, float('nan')):<13.2f}" for c in configs)
        rows.append(f"| {task:<16} | {cells} |")
    return "\n".join(rows)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def transcribe(
    vqa: str = typer.Option(DEFAULT_VQA, help="comma list of VQA corpora"),
    n: int = typer.Option(40, help="items per corpus"),
    out: str = typer.Option("", help="transcripts.jsonl path (default: derived from corpora + n)"),
    device: str = typer.Option("cuda"),
) -> None:
    """GLM-OCR transcribes every corpus image, in its OWN process so the 20GB is freed before the
    decoder loads. One {corpus, idx, transcript} row per item, in load_items() order."""
    vqa_names = [s for s in vqa.split(",") if s]
    items = load_items(vqa_names, n)
    print(f"[transcribe] {len(items)} images", flush=True)
    texts = glm_ocr_transcribe([(img,) for _, img, _, _ in items], device=device)
    out_path = Path(out) if out else _transcript_path(vqa_names, n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, ((corpus, _, _, _), text) in enumerate(zip(items, texts)):
            f.write(json.dumps({"corpus": corpus, "idx": i, "transcript": text}) + "\n")
    print(f"[transcribe] wrote {out_path}", flush=True)


@app.command()
def bakeoff(
    ckpt: str = typer.Argument(..., help="trained glm_ocr adapter checkpoint (alltasks_mb1 best.pt)"),
    vqa: str = typer.Option(DEFAULT_VQA, help="comma list of VQA corpora (default = the 12-task matrix)"),
    n: int = typer.Option(40, help="items per corpus"),
    configs: str = typer.Option(",".join(CONFIGS), help="comma list: encoder, tool, encoder_tool"),
    base: str = typer.Option("poolside/Laguna-XS.2"),
    out: str = typer.Option("results/tool_eval/bakeoff.json", help="where to write the matrix"),
) -> None:
    """encoder vs tool vs encoder_tool over the VQA suite (decoder-controlled). Auto-transcribes first."""
    vqa_names = [s for s in vqa.split(",") if s]
    cfgs = [s for s in configs.split(",") if s]
    matrix = run_bakeoff(ckpt, vqa_names, n, cfgs, base)
    table = _render(matrix, vqa_names, cfgs)
    print("\n" + table, flush=True)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n": n, "configs": cfgs, "matrix": matrix}, indent=2))
    print(f"\nwrote {out_path}", flush=True)


@app.command()
def rescore(
    preds: str = typer.Option("results/tool_eval/preds.jsonl", help="bake-off per-item preds"),
    caption: str = typer.Option("chart2text", help="comma list of free-form caption corpora"),
) -> None:
    """Re-score the caption corpora with ROUGE-L F1. exact/substring `_match` scores a correct paragraph-
    length caption 0 (different surface words) — a measurement artifact. NB the bake-off caps generation at
    24 tokens, so this is a partial-credit floor; a fair caption number also needs longer generation."""
    rows = [json.loads(line) for line in Path(preds).read_text().splitlines() if line.strip()]
    cap = {s for s in caption.split(",") if s}
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["corpus"] in cap:
            agg[(r["config"], r["corpus"])].append(rouge_l_f1(r["gold"], r["pred"]))
    for (config, corpus), scores in sorted(agg.items()):
        print(f"{config:14} {corpus:12} ROUGE-L {sum(scores) / len(scores):.2f}  "
              f"(exact-match 0.00, n={len(scores)})", flush=True)


@app.command("build-docs")
def build_docs(
    keep: str = typer.Option(GLYPH_VQA, help="corpora to keep in the pack (OCR-answerable glyph tasks)"),
    n: int = typer.Option(40, help="items per corpus"),
    out: str = typer.Option("results/tool_eval/loop_docs.jsonl", help="docs-pack path"),
) -> None:
    """Emit a {cat,id,text,q,a} docs pack for the agentic loop: real corpus questions/golds with the
    cached (noisy) GLM-OCR transcript as `text`. Reuses the full-suite transcript cache (no GPU) and
    keeps only the glyph corpora the tool can answer, so the loop runs end-to-end on real OCR noise."""
    full = [s for s in DEFAULT_VQA.split(",") if s]
    transcripts = _ensure_transcripts(full, n, _transcript_path(full, n))
    items = load_items(full, n)
    assert len(transcripts) == len(items), f"{len(transcripts)} transcripts vs {len(items)} items"
    keep_set = {s for s in keep.split(",") if s}
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w") as f:
        for i, (t, (corpus, _img, q, gold)) in enumerate(zip(transcripts, items)):
            if corpus not in keep_set:
                continue
            f.write(json.dumps({"cat": corpus, "id": f"{corpus}_{i}.png",
                                "text": t, "q": q, "a": str(gold)}) + "\n")
            kept += 1
    print(f"[build-docs] kept {kept}/{len(items)} docs ({sorted(keep_set)}) -> {out_path}", flush=True)


if __name__ == "__main__":
    app()
