# Agentic OCR loop on REAL extraction noise

> `feat/agentic-ocr-tool`. Probe: `laguna_rlvr.probe env=ocr_tool_real model=laguna_m1`, hosted
> `poolside/laguna-m.1` ($0), real GLM-OCR transcripts as the `ocr()` backend. Pack:
> `tool_eval build-docs` → `results/tool_eval/loop_docs.jsonl`. Raw: `results/probe/_raw_ocr_tool_real__laguna/`.

## Question

The mock loop ([[baseline]]) proved base Laguna drives the tool loop at **100%** — but under *perfect* OCR,
so it measured only the loop, not extraction. The bake-off ([[bakeoff]]) used real GLM-OCR transcripts but
**single-shot** (inject transcript, take it or leave it), not the agentic loop. The union was never run:
**the agentic multi-turn loop calling a real, noisy GLM-OCR.** That's the realistic RLVR setting — does
real extraction noise break the saturation, and where?

Setup: the same `OCRToolEnv`, but each row is a real glyph-corpus VQA pair whose `ocr()` returns the
**cached GLM-OCR transcript** of the real image (the noisy backend), not hand-written perfect text. 7
glyph corpora × 40 items × 2 rollouts = 560 multi-turn rollouts, native tool-call scaffold, remote
laguna-m.1.

## Result — real noise breaks the 100%, and the drop is corpus-shaped

| corpus | real-OCR loop | mock loop |
|---|---|---|
| textvqa | 0.60 | — |
| ocrvqa | 0.51 | — |
| chartqa | 0.31 | — |
| dvqa | 0.21 | — |
| infographic_vqa | 0.17 | — |
| docvqa | 0.11 | — |
| visualmrc | 0.04 | — |
| **overall** | **0.28** | **1.00** (mock, synthetic docs) |

(The mock baseline ran on 14 synthetic docs with perfect text, so 1.00 → 0.28 is the *regime* change, not
an item-controlled delta.)

**Reading — the loop is intact; the new bottleneck is extraction, and it's per-corpus.** The 0.28 isn't a
loop failure: the model calls `ocr`, reads the transcript, and answers. Success simply tracks **whether
GLM-OCR put the answer in the transcript**:

- **OCR-friendly text wins.** textvqa 0.60, ocrvqa 0.51 — short salient strings GLM-OCR nails (gold
  `nokia` ← OCR `NOKIA`; book authors/titles), so the loop solves them.
- **Chart-value questions partly fail.** chartqa 0.31, dvqa 0.21 — the transcript carries axis labels but
  often drops the bar/point *values* the question needs (`"Title\nValues\nessay\nsoil"` — no numbers), so
  "which bar is largest" is unanswerable from text alone. This is the boundary the *encoder* covers
  (bake-off plotqa 0.75 vs tool 0.00) — confirming the two channels are complementary, not interchangeable.
- **Dense reading-comprehension fails hardest.** visualmrc 0.04, docvqa 0.11, infographic 0.17 — GLM-OCR
  surfaces a header or a fragment but not the specific span (gold `verify-contact-details.namecheap.com`,
  OCR gave only the article title "Checking Your Domain Name Servers…"), and the answer needs reading
  comprehension over text the transcript only partially carries.

**This corpus-shaped spread (0.04 → 0.60) is the RLVR signal.** Under mock OCR it's a flat 1.00 — nothing
to learn. Under real OCR the reward is non-saturated and structured: the learnable behavior is **per-item
trust** — recognising when the transcript is sufficient vs when to re-read, ask differently, or defer.
That's a non-sparse reward (unlike native dense reading, which sits at ≈0), so RLVR has signal to climb.

Side-findings (bug fixes folded in — the loop was impossible on real corpora without them):
1. **The prompt must name the image id.** The mock docs embedded the filename in the question; real corpus
   questions don't, so the model had no handle for `ocr(image_id=…)` and just asked for it ("I need the
   image ID"). The env now states `(id: …)` explicitly. This is a setup fix, not a hint — without the id
   the task can't start.
2. **Terse VQA golds carry trailing periods** (`"Soil."`, `"Land."`) that exact-match `_norm` didn't strip,
   failing correct answers. `_norm` now strips surrounding sentence punctuation (internal dots survive, so
   `42.50` is intact).

## Next

This is the **GO for RLVR**: the loop works, real OCR gives a non-saturated, corpus-shaped reward, and the
headroom is exactly the trustworthy-transcript behavior RLVR can shape. The precondition before training is
the **OCR-backend bake-off by transcription WER** — GLM-OCR's per-corpus extractability (visualmrc 0.04 vs
textvqa 0.60) is the ceiling on loop success, so a better backend on the dense corpora raises the whole
curve. Choose the backend by WER, then train per-item trust on top.
