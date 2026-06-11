# Tool-trace SFT warm-start — teach the adapter to drive the encoder+decoder+tool loop

> Scoping doc. Unblocks `VisionToolEnv` ([[vision-tool-env-floor]]): the SFT adapter answers single-shot
> (`encoder_tool` 0.36) but emits `</assistant>` under a tool prompt — it lacks the *format + multi-turn*
> layer, not the answering capability. Warm-start it on golden tool-call traces, then RLVR.

## The gap (precise)

The `glmocr_alltasks` adapter was SFT'd only on `{IMAGE}\nQ\nAnswer:` → answer. It has **vision grounding**
(the encoder channel works — bake-off `encoder_tool` 0.36) but **no tool-call syntax and no multi-turn
structure**. Probed on the agentic loop it scored 0.00/280, 239/241 replies `</assistant>`. So the warm-start
adds *only the missing layer* — emit `poolside` tool calls across turns — on top of an adapter that can
already read the image and produce the answer. That's a small delta, not a from-scratch capability.

## Training data — golden traces, synthesized (no labels needed)

For each glyph-corpus item `(image, question, gold, transcript)` (the same source `VisionToolEnv` uses —
`load_items` + the Qwen3-VL glyph cache), synthesize a 2-turn trace in the *exact* format `parse_call(...,
"poolside")` expects:

```
[user]  <image> + question + tool instructions       (the VisionToolEnv turn-0 prompt)
[asst]  ocr(image_id=…)                               ← supervised
[user]  [ocr of …] <transcript>                       (observation, NOT supervised)
[asst]  answer(value=<gold>)                           ← supervised
```

**Mix in direct-answer traces** (no `ocr` call: `[asst] answer(value=<gold>)` at turn 0) for items where the
encoder alone suffices — so the model learns *when* to call `ocr` vs answer from vision, not "always call
ocr." That per-item-trust ratio is exactly what RLVR then sharpens ([[ocr-backend-verdict]]: docvqa needs the
transcript, dvqa/charts are the encoder's job). Split ~ by the bake-off's per-corpus encoder-vs-tool edge.

## Objective — multi-turn masked CE

Reuse the adapter's existing machinery: `_embed_multi` (vision splice across turns, already used by `chat`)
+ `_batched_lm_loss` (the QA-SFT batched loss). The only new piece is **multi-turn masking**: build one
sequence `[turn0_prompt + ocr_call + obs + answer_call]`, label = −100 everywhere except the two assistant
spans (`ocr(...)` and `answer(...)`). This is `forward_qa`'s mask pattern extended to two supervised spans
with an unsupervised observation between — a thin generalization of the existing single-span QA-SFT.

## Reuse vs build

| piece | exists | source |
|---|---|---|
| image + q + gold + transcript | ✅ | `load_items` + `ocr_backend_eval._glyph_transcripts` |
| poolside trace format | ✅ | `scaffold.render_instructions` / `parse_call` (the VisionToolEnv prompt) |
| vision splice across turns | ✅ | `model._embed_multi` |
| batched masked-CE loss | ✅ | `model._batched_lm_loss` / `forward_qa` masking |
| warm-start from a checkpoint | ✅ | `load_adapter_state_dict` (as VisionToolEnv/tool_eval do) |
| **trace synthesizer** | ❌ | **build** — `(item) -> (text, supervised_spans)` (~40 lines) |
| **multi-turn SFT step** | ⚠️ | generalize `forward_qa` to two supervised spans (~30 lines) |

## Sequence & validation

1. `synth_traces(glyph, n)` → golden traces (ocr-then-answer + direct-answer mix). Unit-test: every trace
   round-trips through `parse_call` (the model is being taught a format the env can read).
2. `train_tool_sft` — warm-start `glmocr_alltasks` on the traces (projector + LoRA, a few hundred steps).
3. **Re-probe `VisionToolEnv`** on the warm-started ckpt → expect it to *leave the floor* (drive the loop,
   emit parseable calls). Success bar: clears 0.00 and approaches the single-shot `encoder_tool` 0.36; the
   gap above that is the agentic headroom RLVR targets.
4. **RLVR (GSPO, [[architecture-bakeoff-verdict]]'s track)** on the warm-started adapter — shape per-item
   trust (when to trust the transcript vs the grounding). This is where `VisionToolEnv` becomes a learnable
   portfolio member instead of a floored one.

## Risks

- **Trace overfitting** (always-call-ocr) → the direct-answer mix is the mitigation; validate the ocr-call
  *rate* per corpus matches the bake-off's encoder-vs-tool edge, not 100%.
- **Format-only, not grounding** — warm-start teaches syntax; if accuracy doesn't approach 0.36, the loss is
  teaching format without using vision. Guard: the answer span is the *gold*, so CE can only drop by reading
  the image (the QA-SFT no-shortcut property carries over).
- **Cold-start still flat for RLVR?** — warm-start is precisely to lift it off 0.00 so GSPO has gradient.
