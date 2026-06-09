# Agentic OCR-tool — base Laguna loop baseline

> `feat/agentic-ocr-tool`. Probe: `laguna_rlvr.probe env=ocr_tool model=laguna_m1`, hosted
> `poolside/laguna-m.1` ($0), mock perfect-OCR backend. Raw: `results/probe/_raw_ocr_tool__laguna/`.

## Question

The native line is exhausted — the frozen Laguna decoder cannot transcribe dense glyphs from any
representation ([[glm-ocr-encoder-experiment]]). The agentic route sidesteps that: give Laguna an
`ocr(image)->text` tool and let it reason over the extracted text. **Go/no-go: can base Laguna already
drive the tool loop** (decide to call `ocr`, parse the right field, `answer`) over the glyph subset
where the adapter hit the wall — *before* any RLVR? Mock backend returns perfect text, so this isolates
the agent loop + reasoning from OCR extraction noise. `vqav2` etc. are excluded by design: an OCR tool
returns nothing for glyph-independent tasks — those stay on the adapter.

## Result — the loop is saturated (100%)

14 docs across the 5 glyph categories × 3 rollouts = 42 rollouts.

| scaffold | overall | by glyph category |
|---|---|---|
| native (structured tool_calls) | **42/42 (100%)** | docvqa 18/18 · ocrvqa 6/6 · infographic 6/6 · visualmrc 6/6 · chart 6/6 |
| mixed (line/xml/json/poolside) | **42/42 (100%)** | line 12/12 · json 9/9 · poolside 9/9 · xml 12/12 |

0 truncations, ~3 turns/rollout. Verified genuine: e.g. `ocr(image_id="invoice.png")` → reads → `answer(value="42.50")`, correctly picking the total over the subtotal/tax distractors.

**Reading.** Base Laguna-m.1 drives the OCR-tool loop flawlessly, zero-shot, across every glyph category
and every tool-call syntax. The thing the frozen decoder *couldn't* do (transcribe glyphs) is now the
tool's job; the thing Laguna *can* do (reason over text, drive tools) it does at 100%. So **the agent
loop is not the learnable variable** — RLVR has no headroom on it under perfect OCR. The headroom moves
to (a) the **OCR backend** (real extraction noise / partial reads), (b) **harder multi-hop / multi-image**
questions, (c) **knowing when not to trust** a bad OCR and re-reading.

Side-finding (folded into the fix commit): the `mixed` run first read 73.8%, entirely an xml-parser
artifact — Laguna emits a valid `<tool_call>{json}` but reliably drops the closing `</tool_call>`, which
the parser required. Loosening it (`scaffold.py` `_XML_RE`) lifted xml 1/12 → 12/12. The model's intent
was always correct; the parser was strict.

## Next

The mock-OCR loop baseline is a clean **GO**. The interesting baseline is now **end-to-end with a real
OCR backend** (extraction noise) over real corpus images, and a **backend bake-off by transcription WER**
(GLM-OCR vs a general-VLM transcriber vs tesseract — chosen by WER, *not* inherited from the encoder
verdict). Only once the loop meets noisy OCR does RLVR have a learnable signal to train.
