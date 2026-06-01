"""Design2Code — (screenshot, HTML) eval pairs AND the rendered-visual benchmark scorer.

484 real C4 webpages (SALT-NLP/Design2Code, arXiv 2403.03163), disjoint from every training corpus —
so every mixture variant is scored on the *same unseen* set. The HF parquet exposes only the
screenshot, so we fetch the repo snapshot and pair each {id}.png with its {id}.html. Eval only —
never put this in the training mix (leakage).

The scorer follows the Design2Code protocol's five dimensions: the adapter writes HTML from the
screenshot, we render both the generated and reference HTML headless (Playwright), then score
**visual similarity** (the page screenshots, via the adapter's own vision encoder — no extra CLIP
download), plus **block-match / text / position / color** over the rendered visible-text blocks
(bipartite-matched by text). One `d2c/metrics/final` = the mean of the five.
"""
from __future__ import annotations

import io
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import Dataset

from laguna_rlvr.code_exec import extract_code
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter
from laguna_rlvr.visual.multiturn_qa import token_f1

_PROMPT = (f"{IMAGE_TOKEN}\nWrite a single self-contained HTML document (inline CSS) that reproduces "
           "this web page as closely as possible. Output only the HTML.")
_VIEWPORT = (1280, 960)
_METRICS = ("visual_sim", "block_match", "text", "position", "color")
# `final` weights: visual_sim is down-weighted because it saturates (any two web screenshots score
# ~0.95 in encoder space); the structural/content dims carry the score. Sum = 1.0.
_FINAL_WEIGHTS = {"visual_sim": 0.1, "block_match": 0.25, "text": 0.25, "position": 0.2, "color": 0.2}
_LOG_SAMPLES = 8  # sample render-pairs logged to the W&B image-pair table when a run is passed

# Collect each visible text node's content + on-screen bounding box + computed color — the rendered
# "blocks" the position/color/block-match dimensions compare (empty / zero-size nodes are skipped).
_BLOCK_JS = """() => {
  const out = [];
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while (n = w.nextNode()) {
    const t = n.textContent.trim();
    if (!t) continue;
    const r = document.createRange(); r.selectNodeContents(n);
    const b = r.getBoundingClientRect();
    if (b.width === 0 || b.height === 0) continue;
    out.push({text: t, box: [b.left, b.top, b.right, b.bottom],
              color: getComputedStyle(n.parentElement).color});
  }
  return out;
}"""


class Design2Code(Dataset):
    """(screenshot, source-HTML) pairs from the Design2Code repo snapshot."""

    def __init__(self, n: int | None = 128, max_html_chars: int = 8192):
        root = Path(snapshot_download("SALT-NLP/Design2Code", repo_type="dataset"))
        pairs: list[tuple[Path, str]] = []
        for html in sorted(root.glob("*.html")):
            png = html.with_suffix(".png")
            if png.exists():
                pairs.append((png, html.read_text(errors="ignore")[:max_html_chars]))
                if n is not None and len(pairs) >= n:
                    break
        if not pairs:
            raise RuntimeError("no Design2Code {id}.png/{id}.html pairs found in the snapshot")
        self.items = pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Image.Image, str]:
        png, html = self.items[i]
        return Image.open(png).convert("RGB"), html


def _parse_rgb(css: str) -> tuple[int, int, int]:
    """'rgb(34, 34, 34)' / 'rgba(34,34,34,.5)' -> (34, 34, 34); black on anything unparseable."""
    nums = re.findall(r"\d+", css)
    return tuple(int(x) for x in nums[:3]) if len(nums) >= 3 else (0, 0, 0)  # type: ignore[return-value]


def _tokens(s: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _text_f1(gen_blocks: list[dict], ref_blocks: list[dict]) -> float:
    """Token-set F1 over all visible text — does the generated page say the same words as the target."""
    return token_f1(_tokens(" ".join(b["text"] for b in gen_blocks)),
                    _tokens(" ".join(b["text"] for b in ref_blocks)))


def _match_blocks(gen_blocks: list[dict], ref_blocks: list[dict],
                  thresh: float = 0.5) -> list[tuple[dict, dict]]:
    """Greedy one-to-one match of ref<->gen blocks by text similarity (highest-similarity pairs first,
    each block used once, only pairs above `thresh`)."""
    cand = sorted(
        ((SequenceMatcher(None, r["text"], g["text"]).ratio(), gi, ri)
         for ri, r in enumerate(ref_blocks) for gi, g in enumerate(gen_blocks)),
        key=lambda x: x[0], reverse=True)
    used_g: set[int] = set()
    used_r: set[int] = set()
    matches = []
    for sim, gi, ri in cand:
        if sim < thresh or gi in used_g or ri in used_r:
            continue
        used_g.add(gi)
        used_r.add(ri)
        matches.append((gen_blocks[gi], ref_blocks[ri]))
    return matches


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _position(matches: list, ref_diag: float, n_total: int) -> float:
    """Positional accuracy summed over matched blocks but normalized by TOTAL blocks — so unmatched
    blocks count as 0 and it can't read high when only a few blocks matched (the artifact that made the
    baseline's `position` 1.0 at 12% coverage). Each match scores 1 - center-distance/page-diagonal."""
    if not n_total or ref_diag <= 0:
        return float(not n_total)
    score = sum(1 - min(1.0, ((gx - rx) ** 2 + (gy - ry) ** 2) ** 0.5 / ref_diag)
                for g, r in matches
                for (gx, gy), (rx, ry) in [(_center(g["box"]), _center(r["box"]))])
    return score / n_total


def _color(matches: list, n_total: int) -> float:
    """Color accuracy summed over matched blocks, normalized by TOTAL blocks (unmatched -> 0). Each
    match scores 1 - RGB-distance / max(sqrt(3*255^2))."""
    if not n_total:
        return 1.0
    max_d = (3 * 255 ** 2) ** 0.5
    score = sum(1 - sum((a - b) ** 2 for a, b in zip(_parse_rgb(g["color"]), _parse_rgb(r["color"]))) ** 0.5 / max_d
                for g, r in matches)
    return score / n_total


@torch.no_grad()
def _visual_sim(adapter: VisualAdapter, a: Image.Image, b: Image.Image) -> float:
    """Cosine of the two rendered screenshots in the adapter's own vision-encoder space (mean-pooled
    patch features) — a no-extra-dependency stand-in for the protocol's CLIP visual similarity."""
    ea = adapter.encoder.encode([a]).float().mean(dim=1)  # (1, d_enc)
    eb = adapter.encoder.encode([b]).float().mean(dim=1)
    return (F.cosine_similarity(ea, eb).item() + 1) / 2  # map [-1, 1] -> [0, 1]


def _render_blocks(html: str, browser) -> tuple[Image.Image, list[dict]]:
    """Headless-render `html`, return its full-page screenshot + visible text blocks."""
    page = browser.new_page(viewport={"width": _VIEWPORT[0], "height": _VIEWPORT[1]})
    try:
        page.set_content(html, wait_until="networkidle", timeout=15_000)
        png = page.screenshot(full_page=True)
        blocks = page.evaluate(_BLOCK_JS)
    finally:
        page.close()
    return Image.open(io.BytesIO(png)).convert("RGB"), blocks


def _score_pair(adapter: VisualAdapter, gen_img: Image.Image, gen_blocks: list[dict],
                ref_img: Image.Image, ref_blocks: list[dict]) -> dict[str, float]:
    matches = _match_blocks(gen_blocks, ref_blocks)
    ref_diag = (ref_img.size[0] ** 2 + ref_img.size[1] ** 2) ** 0.5
    denom = max(len(gen_blocks), len(ref_blocks))  # block_match = matched fraction of the larger side
    return {"visual_sim": _visual_sim(adapter, gen_img, ref_img),
            "block_match": len(matches) / denom if denom else 1.0,
            "text": _text_f1(gen_blocks, ref_blocks),
            "position": _position(matches, ref_diag, denom),
            "color": _color(matches, denom)}


def design2code_eval(adapter: VisualAdapter, items, n: int | None = None, max_new_tokens: int = 1024,
                     prefix: str = "d2c", run=None, step: int | None = None) -> dict[str, float]:
    """`items`: (screenshot, reference-HTML). The adapter writes HTML from the screenshot; render both
    and score the five Design2Code dimensions. Unrenderable generations score 0 on every dimension.
    With `run`, logs a W&B table of sample reference-vs-generated render pairs for visual inspection."""
    from playwright.sync_api import sync_playwright

    pairs = list(items)[:n] if n is not None else list(items)
    agg: dict[str, list[float]] = defaultdict(list)
    samples: list[tuple] = []  # (ref_img, gen_img, scores) for the W&B image-pair table
    rendered = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for screenshot, ref_html in pairs:
                reply = adapter.chat([Turn(_PROMPT, [screenshot])], max_new_tokens=max_new_tokens)[0]
                gen_html = extract_code(reply) or reply
                ref_img, ref_blocks = _render_blocks(ref_html, browser)
                try:
                    gen_img, gen_blocks = _render_blocks(gen_html, browser)
                except Exception:  # generation didn't render (gibberish / broken markup) -> all-zero
                    for k in _METRICS:
                        agg[k].append(0.0)
                    continue
                rendered += 1
                scores = _score_pair(adapter, gen_img, gen_blocks, ref_img, ref_blocks)
                for k, v in scores.items():
                    agg[k].append(v)
                if len(samples) < _LOG_SAMPLES:
                    samples.append((ref_img, gen_img, scores))
        finally:
            browser.close()
    means = {k: sum(agg[k]) / len(agg[k]) for k in _METRICS if agg[k]}
    out = {f"{prefix}/metrics/{k}": v for k, v in means.items()}
    # weighted mean (renormalized over present dims) — down-weights the saturated visual_sim
    wsum = sum(_FINAL_WEIGHTS[k] for k in means)
    out[f"{prefix}/metrics/final"] = sum(_FINAL_WEIGHTS[k] * v for k, v in means.items()) / wsum if means else 0.0
    # render_rate (like grounding's parse_rate): separates "emitted broken HTML" from "rendered but wrong"
    out[f"{prefix}/metrics/render_rate"] = rendered / len(pairs) if pairs else 0.0
    if run is not None and samples:
        import wandb

        table = wandb.Table(columns=["reference", "generated", "block_match", "text", "visual_sim"])
        for ref_img, gen_img, sc in samples:
            table.add_data(wandb.Image(ref_img), wandb.Image(gen_img),
                           round(sc["block_match"], 3), round(sc["text"], 3), round(sc["visual_sim"], 3))
        run.log({f"{prefix}/samples": table}, step=step)
    return out
