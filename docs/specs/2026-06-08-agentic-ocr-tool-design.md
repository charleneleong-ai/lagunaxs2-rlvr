# Agentic OCR-tool route (RLVR) vs the dense-OCR decoder wall

> 2026-06-08 · `feat/glm-ocr-encoder` · the pivot validated by
> [`docs/experiments/glm-ocr-encoder/results.md`](../experiments/glm-ocr-encoder/results.md).
> **Status: design only — not launched.** Awaiting go-ahead.

## Why pivot (the native door is closed)

The OCR-dense wall (docvqa/ocrvqa/visualmrc ≈ 0) has now survived every native lever: 3 general encoders, an
OCR-native encoder (GLM-OCR, 6045 glyph-level patches) with proper Stage-1 grounding (vqav2 0.61), LoRA rank
64→256, decoder-FFN plasticity (lora-moe / top-k), resolution, and resampler width (which *hurt*). The grounding
control proves the pipeline works; the dense tasks stay dead. **The frozen Laguna decoder cannot transcribe dense
glyphs from any visual representation.** No adapter/encoder change fixes that.

**Why RLVR can work here when it can't on native reading.** RLVR amplifies behaviours the policy already produces
sometimes; it cannot bootstrap from a zero success rate. Native dense reading has success rate ≈ 0 (docvqa 0 final
*and* peak) → a flat-reward desert, no gradient. But **answering a doc question from already-extracted OCR text is a
language task within the decoder's competence** (vqav2 0.61 shows grounding + reasoning work) → non-zero success
rate → real reward signal. The agentic framing routes *around* the wall: the tool reads the glyphs; the decoder
orchestrates and reasons.

## Design — tool-augmented doc-QA, SFT cold-start → RLVR

**Capability taught: orchestration, not transcription.** Give the model an `ocr(image[, region]) -> text` tool. A
turn: see the image → decide to call `ocr` → receive extracted text spliced back into context → answer. The OCR
backend is a real OCR system (GLM-OCR run in its native generate mode, or an off-the-shelf doc-OCR) — *not* the
frozen adapter path.

1. **Tool interface.** A `<tool:ocr>` call (optionally with a bbox/region for targeted reads) → harness runs OCR →
   returns text as a tool message. Reuse the `<image>` splice machinery for any returned crops. Keep the tool
   contract minimal (one tool, string out) to keep the RL action space small.
2. **SFT cold-start (DeepSeek-R1-style).** Generate tool-use traces on the doc family: for each (image, question,
   gold), build a trace `[question] → ocr(image) → [extracted text] → [answer]` (answer = gold, OCR = teacher/real
   OCR). SFT the adapter (+ attn-LoRA) on these traces to lift the tool-call + answer success rate above zero. This
   is the bootstrap RLVR needs.
3. **RLVR sharpen.** GRPO-style on the doc family with a **verifiable reward** = answer correctness vs gold
   (exact-match / normalized-F1 for docvqa/ocrvqa; the harness already has graders). Optional shaping: small bonus
   for a well-formed tool call, penalty for not calling OCR on a dense image then answering wrong. Train the
   adapter + attn-LoRA (decoder stays frozen — we're training the *policy over the tool*, not the reader).

## Success criterion

docvqa / ocrvqa final ≥ **0.40** sustained over ≥2 evals on the isolated doc family (vs the ≈0 native wall), with
`vqav2` (no-tool semantic control) **not regressing** below its native 0.6 — i.e. the tool helps dense reading
without breaking the general path. Then the all-tasks generalization rerun (dilution + sparse-task regression
guard).

## Harness / orchestration

Run under autoresearch `SweepRunner` (active GPU triage + janitor) via the existing
[`scripts/sweep_glm_ocr.py`](../../scripts/sweep_glm_ocr.py) pattern, extended to a `(cold-start SFT) → (RLVR)`
feedback-gated planner (RLVR only if SFT cold-start lifts tool-call success above a floor). Reward grading reuses
the VQA answer-matchers. Detached daemon, PPID=1.

## Risks / open questions

- **Tool-call format reliability.** A frozen decoder + thin LoRA may struggle to emit well-formed tool calls; the
  SFT cold-start is what de-risks this. If it can't learn the call format, escalate decoder plasticity for the
  *orchestration* (not reading) — a much easier target than transcription.
- **OCR backend latency** in the RL loop (each rollout runs OCR). Cache OCR per image; pre-extract the doc-family
  corpus once.
- **Reward hacking.** Exact-match on short answers can be gamed by priors; pair with the `vqav2` no-tool control and
  spot-check traces that the answer actually used the OCR text.
- **Scope.** This is a different training mode (RL + tool harness) than the SFT adapter work — larger build. Start
  with the SFT cold-start alone (cheap) to confirm tool-use lifts docvqa at all before standing up the RL loop.

## Decision after

- docvqa lifts with the tool → the capability is *deliverable* via orchestration; productionize + generalize.
- even tool-augmented docvqa stays low → the bottleneck is the decoder's reasoning over OCR text (not just reading)
  → re-evaluate base-model choice for the doc-QA product.
