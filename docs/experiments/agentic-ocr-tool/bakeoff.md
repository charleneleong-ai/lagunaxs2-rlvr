# Architecture bake-off — encoder vs OCR-tool vs both

> `feat/agentic-ocr-tool`. Harness: `laguna_rlvr.visual.tool_eval bakeoff <ckpt> --n 40`.
> Adapter: `glm_ocr__Laguna-XS.2__mix__glmocr_alltasks_mb1/best.pt` (the all-tasks row).
> Matrix: `results/tool_eval/bakeoff.json` · per-item preds: `results/tool_eval/preds.jsonl`.

## Question

The mock-loop baseline ([[agentic-ocr-tool-baseline]]) showed base Laguna drives the tool loop at 100%
under *perfect* OCR — but that only covers the glyph subset, and it leaves the real question open: **on
the full 12-task VQA suite, does the OCR tool replace the vision encoder, or complement it?** Three
configs, **one decoder, identical items, identical GLM-OCR transcripts** — so the only variable is what
the decoder sees:

| config | prompt | sees |
|---|---|---|
| `encoder` | `{IMAGE}\nQ\nAnswer:` | pixels via the trained adapter, no tool |
| `tool` | `{transcript}\nQ\nAnswer:` | GLM-OCR text only, no image |
| `encoder_tool` | `{transcript}\n{IMAGE}\nQ\nAnswer:` | both |

Decoder-controlled: all three run through the *same* loaded XS.2 adapter weights; `tool` simply skips the
vision splice. GLM-OCR transcription runs in its own subprocess (20GB freed before the 63GB decoder
loads — they can't co-reside in 94GB RAM) and caches to disk, so every config reads the same transcripts.

## Result — the encoder and the tool are complementary; together they dominate

40 items × 12 corpora × 3 configs = 1440 generations.

| task | encoder | tool | **encoder_tool** |
|---|---|---|---|
| **overall** | 0.23 | 0.18 | **0.36** |
| vqav2 | 0.23 | 0.05 | **0.38** |
| visual7w | 0.47 | 0.60 | **0.60** |
| figureqa | 0.90 | 0.72 | **0.90** |
| plotqa | 0.75 | 0.00 | **0.75** |
| dvqa | 0.07 | 0.10 | **0.23** |
| textvqa | 0.03 | 0.10 | **0.33** |
| chartqa | 0.20 | 0.15 | **0.28** |
| chart2text | 0.00 | 0.00 | **0.00** |
| docvqa | 0.03 | 0.12 | **0.15** |
| ocrvqa | 0.00 | 0.28 | **0.45** |
| infographic_vqa | 0.12 | 0.00 | **0.15** |
| visualmrc | 0.00 | 0.00 | **0.15** |

**`encoder_tool` ≥ max(encoder, tool) on every cell**, and wins overall by +13pts over the encoder alone.
The two channels carry different information and the decoder fuses them:

- **The tool covers the glyph wall** where the encoder is flat-zero: `ocrvqa` 0.00 → 0.28, `docvqa`
  0.03 → 0.12, `textvqa` 0.03 → 0.10. This is exactly the dense-glyph reading the frozen decoder
  ([[glm-ocr-encoder-experiment]]) couldn't recover from pixels — now it arrives as text.
- **The encoder covers chart/figure structure** the transcript can't serialise: `plotqa` 0.75 vs **0.00**
  for the tool, `figureqa` 0.90 vs 0.72. A flat OCR dump loses the spatial/axis layout these need.
- **Combined, several cells go super-additive** — `visualmrc` 0/0 → **0.15**, `textvqa` 0.03/0.10 →
  **0.33**, `vqav2` 0.23/0.05 → **0.38**, `ocrvqa` beats both at **0.45**. The grounding and the
  transcript aren't redundant; each rescues items the other misses.
- **`chart2text` stays 0.00 everywhere** — long free-form caption generation that the loosened `_match`
  scores harshly; not an architecture signal.

## Reading

The OCR tool is **not a replacement** for the vision adapter — it's an orthogonal channel. The adapter
grounds spatial/semantic structure (charts, scenes, layout); the tool injects the glyph text the frozen
decoder can't read off pixels. The full-breadth config is the union, and it's the one to train: it never
loses to either single channel and lifts the hardest dense-document tasks off the floor.

Caveat on the encoder baseline: this harness reads `encoder` at 0.23 overall vs 0.34 for the same
checkpoint in [[decoder-unfreeze-experiment]]'s matrix — different item sample (first-40 here) and the
`adapter.chat` template path there vs the gencheck-style `\nAnswer:` prompt here. The **3-way comparison
is internally valid** (one decoder, one prompt convention, one item set); only the cross-doc absolute
number shifts.

## Next

`encoder_tool` is the architecture to carry into RLVR. The learnable signal is no longer the loop (mock
OCR saturates it) nor the encoder-vs-tool choice (settled — use both); it's **OCR-backend noise** and
**knowing when to trust the transcript vs the grounding** on a per-item basis. Backend bake-off by
transcription WER (GLM-OCR vs general-VLM vs tesseract) is the precondition before training.
