"""MMDU (laolao77/MMDU, arXiv 2406.11833) — long-horizon multi-turn multi-image dialogue benchmark.

110 val episodes; each is a long human/gpt dialogue (up to ~12 turn-pairs) over up to ~5-20 images. The
human turns carry `<ImageHere>` placeholders + open-ended questions; the gpt turns are long reference
answers. Images live in `mmdu_pics.zip` (paths in the dialogue are `/mmdu_pics/<name>`); the JSON builder
exposes them only as path strings, so we download + extract the zip once and map paths -> PIL ourselves.
`<ImageHere>` markers are positional: the k-th marker across the whole episode binds the k-th image.

SCORING IS A PROXY, NOT THE OFFICIAL METRIC. Official MMDU is GPT-4 multi-dimensional judging (creativity,
richness, visual perception, ...). Here we use a lightweight reference-overlap proxy for cheap in-loop
tracking: per-turn token-set F1 of the reply vs its gpt reference (`accuracy`), plus a cross-turn memory
proxy (`recall`) — on turns after the first, the fraction whose reply re-surfaces tokens introduced in an
earlier turn's reference (falling back to the final turn vs the first reference when no prior overlap).
"""
from __future__ import annotations

import io
import json
import zipfile

from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from laguna_rlvr.visual.hf_image_text import _cached_or_stream
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter
from laguna_rlvr.visual.multiturn_qa import _norm, token_f1

_REPO = "laolao77/MMDU"
_PLACEHOLDER = "<ImageHere>"
# Stopwords stripped before the recall overlap so memory credit comes from content tokens (entities /
# named text), not function words every long answer shares.
_STOP = frozenset(
    "the a an and or of to in on at is are was were be been it its this that these those for with as by "
    "from into has have had which what when where who how does do their there here image".split()
)


_LOG_EPISODES = 4  # sample episodes whose transcripts are logged to the W&B table when a run is passed


def _tokens(s: str) -> set[str]:
    return set(_norm(s).split())


def _overlap(reply: str, reference: str) -> float:
    """Token-set F1 of `reply` vs `reference` (both `_norm`-folded) — the reference-overlap proxy."""
    return token_f1(_tokens(reply), _tokens(reference))


def _content_tokens(s: str) -> set[str]:
    return _tokens(s) - _STOP


class MMDUDataset(Dataset):
    """MMDU episodes as ordered turn-lists. `__getitem__(i)` -> a list of user-turn dicts
    `{"text": <prompt with IMAGE_TOKEN markers>, "images": [PIL...], "reference": <gpt answer>}`,
    truncated to the first `max_turns` user/assistant pairs."""

    def __init__(self, n: int = 50, max_turns: int = 4):
        self.max_turns = max_turns
        self._raw = _cached_or_stream(f"mmdu__{n}", lambda: self._load(n))
        self._zip = zipfile.ZipFile(hf_hub_download(_REPO, "mmdu_pics.zip", repo_type="dataset"))

    @staticmethod
    def _load(n: int):
        from datasets import Dataset as HFDataset

        path = hf_hub_download(_REPO, "benchmark.json", repo_type="dataset")
        episodes = json.load(open(path))[:n]
        # Persist the dialogue JSON + image-path list verbatim; PIL decode happens lazily per access.
        return HFDataset.from_dict(
            {"conversations": [json.dumps(e["conversations"]) for e in episodes],
             "images": [e["image"] for e in episodes]})

    def _image(self, path: str) -> Image.Image:
        with self._zip.open(path.lstrip("/")) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def _parse(self, conversations: list[dict], image_paths: list[str]) -> list[dict]:
        turns: list[dict] = []
        img_i = 0
        pending: str | None = None  # a user prompt awaiting its assistant reference
        pending_imgs: list[Image.Image] = []
        for msg in conversations:
            if msg["from"] == "user":
                text = msg["value"]
                count = text.count(_PLACEHOLDER)
                pending = text.replace(_PLACEHOLDER, IMAGE_TOKEN)
                pending_imgs = [self._image(image_paths[img_i + k]) for k in range(count)]
                img_i += count
            elif msg["from"] == "assistant" and pending is not None:
                turns.append({"text": pending, "images": pending_imgs, "reference": msg["value"]})
                pending = None
                if len(turns) >= self.max_turns:
                    break
        return turns

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, i: int) -> list[dict]:
        row = self._raw[i]
        return self._parse(json.loads(row["conversations"]), row["images"])


def mmdu_eval(adapter: VisualAdapter, episodes, max_new_tokens: int = 128, prefix: str = "mmdu",
              run=None, step: int | None = None) -> dict[str, float]:
    """Multi-turn dialogue proxy scorer over MMDU `episodes` (each a list of turn dicts from
    `MMDUDataset`). Builds one `Turn` per user turn and runs the whole episode through `adapter.chat`
    (one reply per turn, conditioned on all prior turns + images), then:
      - `{prefix}/metrics/accuracy` = mean per-turn `_overlap(reply, reference)` (reference-overlap proxy)
      - `{prefix}/metrics/recall`   = cross-turn memory proxy: over turns after the first, the fraction
        whose reply re-surfaces content tokens from an EARLIER turn's reference; if no later reply shares
        any such token, fall back to the final reply's overlap with the FIRST turn's reference.

    NOT the official MMDU metric (GPT-4 multi-dimensional judging) — a cheap in-loop overlap proxy.
    With `run`, logs a W&B table of sample conversation transcripts (turn / reply / reference)."""
    accs: list[float] = []
    recalls: list[float] = []
    transcript: list[tuple] = []  # (episode, turn, question, reply, reference, overlap) for the W&B table
    for ep_i, episode in enumerate(episodes):
        if not episode:
            continue
        turns = [Turn(t["text"], t["images"]) for t in episode]
        replies = adapter.chat(turns, max_new_tokens=max_new_tokens)
        refs = [t["reference"] for t in episode]
        accs.extend(_overlap(reply, ref) for reply, ref in zip(replies, refs))
        if ep_i < _LOG_EPISODES:
            transcript.extend((ep_i, ti, t["text"], reply, ref, round(_overlap(reply, ref), 3))
                              for ti, (t, reply, ref) in enumerate(zip(episode, replies, refs)))
        if len(episode) < 2:
            continue
        prior: set[str] = set(_content_tokens(refs[0]))
        hits = 0
        for reply, ref in zip(replies[1:], refs[1:]):
            hits += bool(_content_tokens(reply) & prior)
            prior |= _content_tokens(ref)
        recalls.append(hits / (len(episode) - 1) if hits else _overlap(replies[-1], refs[0]))
    if run is not None and transcript:
        import wandb

        table = wandb.Table(columns=["episode", "turn", "question", "reply", "reference", "overlap"])
        for ep_i, ti, q, reply, ref, ov in transcript:
            table.add_data(ep_i, ti, q[:500], reply[:500], ref[:500], ov)
        run.log({f"{prefix}/transcripts": table}, step=step)
    return {f"{prefix}/metrics/accuracy": sum(accs) / max(len(accs), 1),
            f"{prefix}/metrics/recall": sum(recalls) / max(len(recalls), 1)}
