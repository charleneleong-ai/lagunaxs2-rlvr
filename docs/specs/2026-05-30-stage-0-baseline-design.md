# Stage 0 — Baseline panel (out-of-box Laguna, no adapter)

**Status:** design, awaiting review · **Branch:** `feat/mm-adapter-stage0` · **Date:** 2026-05-30

## Goal

Quantify what **text-only Laguna XS.2** does on the visual coding tasks *before any projector/adapter
training*, so the adapter's value is measurable and we know whether climbing the ladder is justified.
This is Stage 0 of the training ladder ("the bar the adapter must beat"). Eval-only — no training, the
frozen base LLM throughout. Supersedes the Stage-2 agentic-SFT track for now (deferred, not deleted).

## Two axes, two tables

### Axis A — single-turn generation (real corpora)

Run on the real held-out corpora, scored with the existing metric suite:

| Corpus | kind | task prompt | metrics that fire |
|---|---|---|---|
| `design2code` | html | "Write the HTML/CSS that renders this page." | cer, wer, code_valid |
| `chartmimic` | python | "Write the matplotlib code for this chart." | cer, wer, code_valid, **codebleu** |
| `swebench_mm` | (issue-text) | "Describe the software issue shown." | cer, wer |

Two baselines, both the **frozen base Laguna**, no projector:

- **blind** — task prompt only, no visual signal. The floor: how much is solvable from text alone.
- **tool-mediated (OCR→Laguna)** — GLM-OCR transcribes the image → transcript + task prompt → Laguna.
  The realistic no-adapter agent path = the bar the adapter must beat.

The trained adapter later slots in as a third row via the existing `eval.py` path → one apples-to-apples
table per corpus.

### Axis B — multi-turn multimodal QA, grounded in our data mixture (persisted)

A *different* capability axis from A: reading + vision-as-tool-observation persisting across turns. The
existing `multiturn_qa.py` reads toy `render_text("invoice 5")` images — that measures nothing about the
real task. **Build episodes from the real mixture corpora instead** (the same screenshots/charts the model
will actually face), keeping deterministic scoring via a **needle extracted from the paired label**:

| corpus kind | needle extractor (from the label) | turn-1/2 question |
|---|---|---|
| python (chartmimic) | chart title via `set_title("…")` / `plt.title("…")` | "What is the title of this chart?" |
| html (design2code/websight) | `<title>…</title>`, else first `<h1>` text | "What is the page's title/heading?" |

3-turn episode: read real image A → read real image B → text-only "what was the {title} of the first
image?" → recall A's needle. Scored `qa/metrics/accuracy` (per-turn reading) + `qa/metrics/recall`
(cross-turn memory) by substring (the reply is verbose; substring avoids CER over-penalizing). Corpora
whose label has no clean needle (`swebench_mm` prose) are excluded from QA; the toy SyntheticOCR episode
stays available as a controlled sanity check, not the headline benchmark.

- **Persist** a fixed episode manifest to `data/multiturn_qa.jsonl` — per episode: `(corpus_a, idx_a,
  needle_a)`, `(corpus_b, idx_b, needle_b)`, the recall question. Images are re-fetched by
  `load_text_image(corpus)[idx]` (HF dataset row order is stable), so the manifest is tiny, inspectable, and
  decoupled from generator edits. `evaluate_multiturn_qa` loads the manifest if present, else builds (pick
  needle-bearing rows from the mixture) + writes it. A per-kind `extract_needle(label, kind)` is the only
  new logic; it lives next to `CORPUS_KIND` in `corpora.py`.
- Baselines: **blind** (chat with no images → floor), **tool-mediated** (GLM-OCR transcribes each turn's
  image → text multi-turn chat on transcripts), **adapter** (existing vision-splice path, slots in later).

## Architecture — one new module + two reuse refactors

1. **Refactor `metrics.py`** — extract the adapter-free scoring core
   `score_predictions(preds, refs, kinds, prefix="val") -> dict` (pure: wer/cer/code_valid/codebleu, the
   `{prefix}/metrics/*` keys). `generation_metrics` becomes: transcribe via the adapter → call
   `score_predictions`. No metric logic duplicated; baselines call `score_predictions` on their own preds.
2. **Refactor `eval.py`** — extract GLM-OCR image→text into a reusable
   `glm_ocr_transcribe(items, device, max_new_tokens) -> list[str]` (today inlined in `_glm_baseline_cer`).
   Both `eval.py` and the baseline harness call it.
3. **New `visual/baseline.py`** — the harness:
   - `TASK_PROMPT: dict[str|None, str]` keyed by corpus kind (html/python) + a default for swebench.
   - `text_generate(llm, tok, prompts) -> preds` and `text_chat(llm, tok, turn_texts) -> replies` — base-LLM
     text-only generation (no vision splice), the blind/OCR engine for both axes.
   - **Staged GPU** to avoid co-residence of two 30B+/multi-GB models: GLM-OCR transcribes all images first
     → cache transcripts → free GLM-OCR → load base Laguna → run blind + OCR for both axes →
     `score_predictions`.
   - typer CLI: `baseline --dataset design2code --baselines blind,ocr --n-eval 64 [--qa]`. Prints a RESULT
     table (corpus × baseline × metrics) + optional W&B under `baseline/<name>/metrics/*`.

No change to the training loop, the projector, or the frozen-backbone load path.

## Test plan

- `score_predictions` parity — same keys/values as the pre-refactor `generation_metrics` on a fixed
  (preds, refs, kinds) triple.
- `TASK_PROMPT` returns a non-empty instruction for html/python/None kinds.
- `glm_ocr_transcribe` shape (mocked model) — N images → N strings.
- `extract_needle` — pulls the title from a `set_title("…")` snippet and a `<title>`/`<h1>` HTML snippet;
  returns None when absent (so that row is skipped, not silently mis-scored).
- Manifest round-trip — build→write→load yields identical episodes; loader re-fetches the right corpus rows.
- blind/OCR QA variants return `{qa/metrics/accuracy, qa/metrics/recall}` in [0,1].
- On-GPU run is the real validation (one A100-80GB, BF16 Laguna + staged GLM-OCR).

## Out of scope / deferred

- VLM-alone ceiling (a second large vision model end-to-end) — skipped per scoping.
- Stage-2 agentic SFT (multi-turn masked training loss) — deferred; this baseline tells us how big the
  gap is first.
- Render-diff / execution metrics — Stage-3 sandbox (unchanged roadmap).
