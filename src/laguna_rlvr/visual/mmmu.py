"""MMMU + MathVista — image+question -> answer accuracy benchmarks (multimodal QA / reasoning).

Both are held-out eval sets scored the same way: splice the image, ask the question, parse the
adapter's reply, compare to gold. MMMU (MMMU/MMMU, arXiv 2311.16502) is college-level multi-discipline
multiple-choice/open QA across per-subject configs (validation split). MathVista
(AI4Math/MathVista, arXiv 2310.02255, testmini split) is visual mathematical reasoning, multi-choice or
free-form. Streamed + disk-cached via the shared `_cached_or_stream` so reruns are offline.

Choice parsing is deliberately lenient (the adapter replies verbosely): a multiple-choice prediction is
read as the first standalone option letter OR a substring match of the option text; free-form / open
answers normalize-match (numeric tolerance, then substring). Each scorer returns one
`f"{prefix}/metrics/accuracy"`.
"""
from __future__ import annotations

import ast
import re
from itertools import islice

from datasets import Dataset as HFDataset
from datasets import Features
from datasets import Image as HFImage
from datasets import Sequence, Value, load_dataset
from rich.progress import track
from torch.utils.data import Dataset

from laguna_rlvr.visual.hf_image_text import _cached_or_stream
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter

# A representative spread of MMMU subjects — the combined `lmms-lab/MMMU` default config is preferred
# (one stream), but MMMU/MMMU is per-subject, so we iterate these and cap to `n` total.
_MMMU_SUBJECTS = ["Accounting", "Art", "Biology", "Computer_Science", "Economics", "History",
                  "Math", "Physics", "Psychology"]
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # full alphabet: MMMU has questions with >7 options


# ── shared parsing ──────────────────────────────────────────────────────────────────────────────────

def _norm_answer(s: str) -> str:
    """Lowercase, strip surrounding punctuation/whitespace, drop a leading currency/percent symbol —
    the normalized form for free-form string comparison (numeric tolerance is handled separately)."""
    return re.sub(r"\s+", " ", str(s).lower().strip().strip("$%.,:;!?\"'()")).strip()


def _as_float(s: str) -> float | None:
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s).replace("$", "").replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def _parse_choice(reply: str, n_options: int) -> str | None:
    """First standalone option letter in `reply` within the first `n_options` choices — matches
    'A', '(B)', 'B.', 'The answer is C'. Returns the uppercase letter, or None if absent."""
    valid = _LETTERS[:n_options]
    m = re.search(rf"\b([{valid}])\b", reply.upper())
    return m.group(1) if m else None


def _match_choice(reply: str, choices: list[str]) -> str | None:
    """Match `reply` to one of `choices` by substring of the choice text OR by option letter
    (A->choices[0]). Substring wins — naming the option is stronger evidence than a bare letter, which
    a standalone article 'a' in prose would otherwise spuriously trip. Returns the choice, or None."""
    r = _norm_answer(reply)
    for c in choices:
        if (nc := _norm_answer(c)) and nc in r:
            return c
    if letter := _parse_choice(reply, len(choices)):
        return choices[_LETTERS.index(letter)]
    return None


def _free_match(reply: str, gold: str) -> bool:
    """Free-form match: numeric within tolerance if both parse as numbers, else normalized substring."""
    rf, gf = _as_float(reply), _as_float(gold)
    if rf is not None and gf is not None:
        return abs(rf - gf) <= 1e-2 * max(1.0, abs(gf))
    g = _norm_answer(gold)
    return bool(g) and g in _norm_answer(reply)


def _option_lines(options: list[str]) -> str:
    return "".join(f"\n{_LETTERS[i]}. {opt}" for i, opt in enumerate(options))


# ── MMMU ────────────────────────────────────────────────────────────────────────────────────────────

class MMMUDataset(Dataset):
    """MMMU multi-discipline QA: (image, question+options, answer_letter, question_type). Streams a
    sampling across `subjects` (per-subject configs), capped to `n` total; uses only the first image."""

    def __init__(self, n: int = 300, subjects: list[str] | None = None, split: str = "validation"):
        subjects = subjects or _MMMU_SUBJECTS
        key = "mmmu__" + "__".join(str(p) for p in (n, split, *subjects))
        self._ds = _cached_or_stream(key, lambda: self._stream(n, subjects, split))

    @staticmethod
    def _stream(n: int, subjects: list[str], split: str) -> HFDataset:
        imgs, qs, ans, qtypes = [], [], [], []
        per = max(1, n // len(subjects))
        for subject in subjects:
            if len(imgs) >= n:
                break
            stream = load_dataset("MMMU/MMMU", subject, split=split, streaming=True)
            for row in islice(stream, min(per, n - len(imgs))):
                img, q, answer = row.get("image_1"), row.get("question"), row.get("answer")
                if img is None or not q or not answer:
                    continue
                qtype = row.get("question_type") or "multiple-choice"
                options = _parse_options(row.get("options"))
                question = q + _option_lines(options) if options else q
                imgs.append(img.convert("RGB"))
                qs.append(question)
                ans.append(answer)
                qtypes.append(qtype)
        if not imgs:
            raise RuntimeError("no usable MMMU rows")
        return HFDataset.from_dict(
            {"image": imgs, "question": qs, "answer": ans, "question_type": qtypes},
            features=Features({"image": HFImage(), "question": Value("string"),
                               "answer": Value("string"), "question_type": Value("string")}))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        row = self._ds[i]
        return row["image"], row["question"], row["answer"], row["question_type"]


def _parse_options(raw) -> list[str]:
    """MMMU `options` is a stringified python list ('["x", "y"]'); parse to a list of option strings."""
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        val = ast.literal_eval(raw)
        return list(val) if isinstance(val, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def mmmu_eval(adapter: VisualAdapter, items, max_new_tokens: int = 64,
              prefix: str = "mmmu") -> dict[str, float]:
    """`items`: (image, question_with_options, answer_letter, question_type). Multiple-choice scores by
    parsed option letter vs gold; open questions by substring of the gold. Returns accuracy."""
    hits = total = 0
    for image, question, answer, qtype in items:
        reply = adapter.chat([Turn(f"{IMAGE_TOKEN}\n{question}", [image])], max_new_tokens=max_new_tokens)[0]
        if str(qtype).lower().startswith("open"):
            ok = _free_match(reply, answer)
        else:
            n_opt = sum(1 for ln in question.splitlines() if re.match(r"^[A-G]\.\s", ln)) or len(_LETTERS)
            ok = _parse_choice(reply, n_opt) == str(answer).strip().upper()
        hits += int(ok)
        total += 1
    return {f"{prefix}/metrics/accuracy": hits / max(total, 1)}


# ── MathVista ─────────────────────────────────────────────────────────────────────────────────────────

class MathVistaDataset(Dataset):
    """MathVista visual-reasoning QA: (image, query_or_question, answer, question_type, choices). Prefers
    the dataset's prompt-formatted `query` field, falling back to `question`. `choices` is a list or []."""

    def __init__(self, n: int = 300, split: str = "testmini"):
        key = "mathvista__" + "__".join(str(p) for p in (n, split))
        self._ds = _cached_or_stream(key, lambda: self._stream(n, split))

    @staticmethod
    def _stream(n: int, split: str) -> HFDataset:
        stream = load_dataset("AI4Math/MathVista", split=split, streaming=True)
        imgs, prompts, ans, qtypes, choices = [], [], [], [], []
        for row in track(islice(stream, n), total=n, description=f"MathVista ({n})"):
            img = row.get("decoded_image") or row.get("image")
            prompt = row.get("query") or row.get("question")
            answer = row.get("answer")
            if img is None or not prompt or answer is None:
                continue
            imgs.append(img.convert("RGB"))
            prompts.append(prompt)
            ans.append(str(answer))
            qtypes.append(row.get("question_type") or "free_form")
            choices.append(list(row.get("choices") or []))
        if not imgs:
            raise RuntimeError("no usable MathVista rows")
        return HFDataset.from_dict(
            {"image": imgs, "question": prompts, "answer": ans,
             "question_type": qtypes, "choices": choices},
            features=Features({"image": HFImage(), "question": Value("string"),
                               "answer": Value("string"), "question_type": Value("string"),
                               "choices": Sequence(Value("string"))}))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        row = self._ds[i]
        return row["image"], row["question"], row["answer"], row["question_type"], row["choices"]


def mathvista_eval(adapter: VisualAdapter, items, max_new_tokens: int = 64,
                   prefix: str = "mathvista") -> dict[str, float]:
    """`items`: (image, prompt, answer, question_type, choices). `multi_choice` matches the predicted
    choice (by letter or choice-text substring) against the gold choice; `free_form` normalized-matches
    (numeric tolerance). Returns accuracy."""
    hits = total = 0
    for image, prompt, answer, qtype, choices in items:
        reply = adapter.chat([Turn(f"{IMAGE_TOKEN}\n{prompt}", [image])], max_new_tokens=max_new_tokens)[0]
        if str(qtype).startswith("multi") and choices:
            ok = _match_choice(reply, list(choices)) == str(answer)
        else:
            ok = _free_match(reply, answer)
        hits += int(ok)
        total += 1
    return {f"{prefix}/metrics/accuracy": hits / max(total, 1)}
