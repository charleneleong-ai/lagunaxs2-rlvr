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

## Next

RLVR runs on the **Qwen3-VL** transcript backend. The remaining headroom is per-item trust (when to re-read
vs defer) plus two levers this run surfaced but didn't pull: a **longer transcription budget** for dense
documents, and the **encoder channel** for chart-value questions the tool structurally can't answer.
