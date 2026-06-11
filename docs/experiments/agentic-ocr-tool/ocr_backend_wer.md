# OCR-backend bake-off — GLM-OCR vs Qwen3-VL by extraction quality

> `feat/ocr-backend-wer`. Harness: `laguna_rlvr.visual.ocr_backend_eval bakeoff --n 40`.
> Result: `results/ocr_backend/bakeoff.json` · transcripts: `results/ocr_backend/<backend>__<probe>__n40.jsonl`.

## Question

The real-OCR loop ([[real_ocr_loop]]) showed loop success tracks **whether the OCR backend put the answer
in the transcript** — GLM-OCR's per-corpus extractability (visualmrc 0.04 … textvqa 0.60) is the *ceiling*
on loop success; the decoder can't answer what the transcript never captured. So before RLVR the open
question is upstream of the decoder entirely: **would a different OCR backend raise that ceiling, and
where?** This bake-off strips the decoder out and compares two backends — `glm_ocr` (OCR-native) vs
`qwen3_vl` (Qwen3-VL-8B, general VLM) — on the transcript alone, two ways:

| axis | set | measures |
|---|---|---|
| **answer-coverage** | 7 glyph VQA corpora | fraction of items whose gold answer is recoverable from the transcript (the loop's own `_match`) — reference-free, = the per-corpus ceiling on loop success |
| **WER / CER** | `cauldron_rendered_text` | literal word/char error vs the rendered-text reference (the label *is* the visible text) — decoder-independent extraction fidelity |

Each backend transcribes in its own subprocess (memory isolation — the two VLMs never co-reside), under the
same 48-token transcription budget; the parent scores on CPU. WER/CER reuse the repo's jiwer-backed
`metrics.{wer,cer}` after case/whitespace normalization.

## Result — Qwen3-VL wins both axes; the lift is concentrated where extraction was the bottleneck

| corpus | glm_ocr | **qwen3_vl** |
|---|---|---|
| **overall** | 0.21 | **0.25** |
| textvqa | 0.68 | 0.68 |
| ocrvqa | 0.45 | **0.47** |
| chartqa | **0.23** | 0.20 |
| docvqa | 0.12 | **0.35** |
| dvqa | 0.00 | 0.00 |
| infographic_vqa | 0.03 | 0.03 |
| visualmrc | 0.00 | 0.00 |

| backend | WER ↓ | CER ↓ |
|---|---|---|
| glm_ocr | 0.051 | 0.038 |
| **qwen3_vl** | **0.034** | **0.028** |

**Qwen3-VL is the better backend: lower transcription error (WER −33% rel.) AND higher coverage (0.25 vs
0.21).** But the two axes decouple, and reading them together is the actual finding:

- **The coverage lift is almost entirely docvqa (0.12 → 0.35).** Dense business documents are exactly where
  extraction fidelity is the bottleneck, and Qwen reads more of the page into the same 48-token budget. Two
  cached examples: gold `0.28` — GLM truncates at the chart header (`FIGURE C.2 … CANADA … PER 1000`) before
  reaching the value; Qwen reaches it (`… CANADA / 0.3 / 0.28 / PER / 1000`). Gold `itc limited` — GLM starts
  mid-page (`ITC's Brands: An Asset…`); Qwen captures the cover header `ITC Limited REPORT AND ACCOUNTS 2013`.
- **The floored corpora don't move — because their floor is NOT extraction fidelity.** dvqa 0/0, visualmrc 0/0,
  infographic 0.03/0.03 stay flat under the better backend. The bottleneck there is structural, not OCR
  quality: dvqa needs chart *bar values* a flat transcript can't serialise (the **encoder's** job — bake-off
  [[bakeoff]] plotqa: tool 0.00 vs encoder 0.75); visualmrc/infographic golds are comprehension phrases that
  aren't verbatim in the page (`"The first example message shown is 'verify-contact-details.namecheap.com'"`),
  so no transcript — however clean — substring-matches them. A better OCR backend cannot lift a floor that
  isn't about reading the glyphs.
- **WER on clean text ≠ coverage on dense images.** Both backends transcribe `cauldron_rendered_text` near-
  perfectly (3–5% WER) — character-level OCR on clean rendered text is essentially solved for both. So the
  coverage gaps on dense VQA are not a clean-text-accuracy problem; they are about dense-layout extraction
  (where Qwen wins) and answer-shape (where neither can help). The two axes measure different difficulty
  regimes and must be read together, not collapsed to one "OCR quality" number.

## Reading

Switch the loop's OCR backend to **Qwen3-VL**: it strictly dominates GLM-OCR on extraction fidelity and lifts
the one corpus (docvqa) whose ceiling was genuinely extraction-bound, at no cost elsewhere. But the backend
swap is **not** a cure for the dense-document wall — visualmrc/infographic/dvqa stay floored because their
bottleneck is the vision encoder (chart values) or the metric/task shape (comprehension answers), the levers
the architecture bake-off already isolated. The backend choice and the encoder-vs-tool complementarity are
orthogonal; this run settles the former.

Caveats: the 48-token transcription budget caps dense-document coverage for both backends — a longer budget
is the obvious next lever on docvqa/visualmrc (analogous to the caption-decoding cap in [[bakeoff]]). And the
WER reference is clean rendered text (easier than the dense VQA images), so 3–5% is a fidelity floor on the
easy regime, not on the dense corpora the loop actually struggles with.

## Loop validation — wiring Qwen3-VL in

The bake-off measures the transcript decoder-free; the test is whether a better transcript moves the *loop*.
Rebuilt the docs pack on Qwen3-VL (`ocr_backend_eval build-docs --backend qwen3_vl`) and re-ran the real-OCR
loop ([[real_ocr_loop]]) item-for-item — same 7 corpora × 40 × 2 rollouts, remote laguna-m.1.

| corpus | GLM loop | Qwen loop | Δ |
|---|---|---|---|
| textvqa | 0.60 | 0.61 | +0.01 |
| ocrvqa | 0.51 | 0.53 | +0.02 |
| chartqa | 0.31 | 0.31 | +0.00 |
| **docvqa** | 0.11 | **0.24** | **+0.13** |
| **dvqa** | 0.21 | **0.09** | **−0.12** |
| infographic_vqa | 0.17 | 0.12 | −0.05 |
| visualmrc | 0.04 | 0.04 | −0.00 |
| **overall** | **0.28** | **0.28** | **−0.00** |

**The predicted extraction-bound lift landed — and only there.** docvqa, the one corpus the bake-off flagged
as extraction-bound (coverage 0.12 → 0.35), moves in the loop by the predicted direction and rough magnitude:
**0.11 → 0.24 (+0.13)**. The coverage proxy correctly predicted the corpus that would move.

**But overall is flat (0.28), because dvqa regresses −0.12.** On dvqa the answer is a chart *value* absent from
**both** transcripts (bake-off coverage 0/0) — loop success there is the decoder guessing from the chart labels
the transcript *does* carry, and Qwen's differently-worded transcript shifts those guesses the wrong way. It is
not a length effect (dvqa transcripts: GLM 13.9 vs Qwen 12.7 mean words). At 40 items/corpus the per-corpus
deltas are suggestive, not definitive, but the direction is clean: **a better transcriber helps where the answer
is textual (docvqa) and can mildly hurt where it isn't (dvqa).**

The lesson sharpens the bake-off itself: coverage/WER predicted the corpus that moved (docvqa) but was **blind
to the one that regressed** (dvqa, coverage 0/0 for both — the proxy can't see a transcript actively misleading
on an answer it never contained). A backend can only be fully judged **in-loop**, not by coverage alone.

## Next

Keep **Qwen3-VL** as the RLVR backend: it strictly wins extraction fidelity (WER) and the one extraction-bound
loop corpus (docvqa) at no overall cost. But the flat overall is the real signal — the backend swap is
**necessary, not sufficient**. The floored/regressing corpora (dvqa, visualmrc, infographic) need the two levers
this run isolated but didn't pull: the **encoder channel** for non-textual answers (chart values) the tool
structurally can't carry, and **per-item trust** — rely on the transcript when it's authoritative (docvqa),
ignore it when the answer isn't in it (dvqa). That trust signal is exactly what RLVR can shape.
