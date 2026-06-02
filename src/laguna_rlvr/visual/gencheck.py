"""One generalizable free-generation check for any adapter checkpoint — replaces the per-run
gencheck_*.py scratch copies (each was a sed of encoder/ckpt/anchor/projector). Builds the adapter from
CLI args, loads the checkpoint, and dumps TRUE vs generated ANSWER across the reading corpora + VQA
suite with the loosened match + embed_norm. Inspect real free-generation — never trust teacher-forced
val loss alone (it is anti-correlated with reading).

    uv run python -m laguna_rlvr.visual.gencheck <ckpt> --encoder siglip_naflex --no-anchor
    uv run python -m laguna_rlvr.visual.gencheck <ckpt> --encoder qwen3_vl_8b --no-anchor --unfreeze lora
"""
from __future__ import annotations

import torch
import typer

from laguna_rlvr.visual.corpora import load_text_image, extract_needle, load_vqa, read_question
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import IMAGE_TOKEN, VisualAdapter
from laguna_rlvr.visual.multiturn_qa import _match

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    ckpt: str,
    encoder: str = "siglip",
    projector: str = "resampler",
    base: str = "poolside/Laguna-XS.2",
    anchor: bool = typer.Option(True, "--anchor/--no-anchor", help="match training's anchor setting"),
    unfreeze: str = typer.Option("", help="'' | lora — must match the checkpoint"),
    pool: int = typer.Option(0, help="encoder pool (0 = auto: 4 for qwen, 1 otherwise)"),
    vqa: str = "textvqa,chartqa,docvqa,ocrvqa",
    n: int = typer.Option(2, help="samples per corpus"),
    max_new_tokens: int = 24,
) -> None:
    pool = pool or (4 if "qwen" in encoder else 1)
    a = VisualAdapter(load_encoder(encoder, pool=pool), base, projector_kind=projector,
                      use_anchor=anchor, unfreeze=unfreeze)
    a.load_adapter_state_dict(torch.load(ckpt, map_location=a.llm.device))
    a.eval()
    print("embed_norm:", round(a.embedding_norm_ratio([load_text_image("synthetic", 1)[0][0]]), 3), flush=True)

    @torch.no_grad()
    def ask(img, q: str) -> str:
        inp = a._embed_with_vision(f"{IMAGE_TOKEN}\n{q}\nAnswer:", a._project([img])[0:1])
        g = a.llm.generate(inputs_embeds=inp, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3)
        return a.tok.decode(g[0], skip_special_tokens=True).split("\n")[0].strip()

    probes = []  # (corpus, image, question, true_answer)
    for c in ("synthetic", "webcode2m", "websight", "design2code"):
        ds = load_text_image(c, n + 2)
        for i in range(n):
            true = ds[i][1] if c == "synthetic" else extract_needle(ds[i][1], "html")
            if true:
                probes.append((c, ds[i][0], read_question(None if c == "synthetic" else "html"), true))
    for name in (s for s in vqa.split(",") if s):
        vq = load_vqa([name], n + 2)[0][0]
        for i in range(n):
            probes.append((name, vq[i][0], vq[i][1], vq[i][2]))

    for c, img, q, true in probes:
        ans = ask(img, q)
        flag = "HIT" if _match(true, ans) else "   "
        print(f"[{flag}|{c:11}] TRUE={str(true)[:32]!r:36} -> {ans[:48]!r}", flush=True)


if __name__ == "__main__":
    app()
