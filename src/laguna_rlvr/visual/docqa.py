"""OCR -> Laguna document/UI QA pipeline (no adapter, no local Laguna weights).

The text-only adapter route needs Laguna's local weights (-> 80GB GPU). This pipeline instead
gives Laguna document understanding via a *tool*: GLM-OCR (local, fits a 40GB card) transcribes a
doc/UI image, and Laguna (Prime hosted inference, $0) answers a question over that transcription.

Self-verifying data: each image is rendered from known fields, so the gold answer is exact — no
dataset download needed. This also measures the baseline the visual adapter must beat.

  python -m laguna_rlvr.visual.docqa --n 8
"""
from __future__ import annotations

import argparse
import random
import re
import subprocess

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

_OCR_REPO = "zai-org/GLM-OCR"
_LAGUNA = "poolside/laguna-xs.2"

# (label, value-generator, question) templates for a synthetic document/UI form.
_FIELDS = [
    ("Invoice No", lambda r: f"{r.randint(10000, 99999)}", "What is the Invoice No?"),
    ("Total", lambda r: f"${r.randint(1, 9)},{r.randint(100, 999)}.{r.randint(10, 99)}", "What is the Total?"),
    ("Date", lambda r: f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}", "What is the Date?"),
    ("Status", lambda r: r.choice(["PAID", "PENDING", "OVERDUE"]), "What is the Status?"),
    ("Account", lambda r: f"AC-{r.randint(1000, 9999)}", "What is the Account?"),
]


def make_items(n: int, seed: int = 0) -> list[dict]:
    """Each item: a rendered multi-field image + one question targeting a field + its exact value."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        fields = [(label, gen(rng), q) for label, gen, q in _FIELDS]
        target = rng.randrange(len(fields))
        label, value, question = fields[target]
        items.append({"lines": [f"{lbl}: {val}" for lbl, val, _ in fields],
                      "question": question, "answer": value})
    return items


def render(lines: list[str], size: tuple[int, int] = (480, 220)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for row, line in enumerate(lines):
        draw.text((16, 16 + row * 34), line, fill="black", font=font)
    return img


@torch.no_grad()
def glm_ocr_transcribe(model, proc, image: Image.Image) -> str:
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": "Transcribe all text in this image."}]}]
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = proc(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    gen = model.generate(**batch, max_new_tokens=128, do_sample=False)
    trimmed = gen[:, batch["input_ids"].shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def ask_laguna(ocr_text: str, question: str) -> str:
    prompt = (f"You are reading the OCR text of a document:\n\n{ocr_text}\n\n"
              f"Question: {question}\nReply with only the value, nothing else.")
    out = subprocess.run(["prime", "inference", "chat", _LAGUNA, prompt, "--plain"],
                         capture_output=True, text=True, timeout=120).stdout
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("Waiting for")]
    return "\n".join(lines).replace("</assistant>", "").strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(_OCR_REPO, device_map=device).eval()
    proc = AutoProcessor.from_pretrained(_OCR_REPO)

    items = make_items(a.n, a.seed)
    correct = 0
    for i, item in enumerate(items):
        ocr = glm_ocr_transcribe(model, proc, render(item["lines"]))
        ans = ask_laguna(ocr, item["question"])
        hit = _norm(item["answer"]) in _norm(ans)
        correct += hit
        print(f"[{i}] Q={item['question']!r} gold={item['answer']!r} "
              f"laguna={ans[:40]!r} {'OK' if hit else 'MISS'}", flush=True)
    print(f"\nDoc-QA accuracy: {correct}/{a.n} = {correct / a.n:.2%}", flush=True)


if __name__ == "__main__":
    main()
