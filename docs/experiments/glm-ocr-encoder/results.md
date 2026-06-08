# GLM-OCR encoder — encoder-wall vs decoder-wall (the decider)

> `feat/glm-ocr-encoder`. Source: W&B `chaleong/laguna-mm-adapter`, runs 2026-06-08.
> Spec: [`docs/specs/2026-06-07-resampler-bottleneck-ocr-wall-design.md`](../../specs/2026-06-07-resampler-bottleneck-ocr-wall-design.md) (the
> finding that motivated this) → next: [`docs/specs/2026-06-08-agentic-ocr-tool-design.md`](../../specs/2026-06-08-agentic-ocr-tool-design.md).

## Hypothesis

Every prior lever left the OCR-dense wall (docvqa/ocrvqa/visualmrc = 0) intact: encoder swaps (SigLIP-NaFlex /
SigLIP-AnyRes / Qwen3-VL), LoRA rank 64→256, decoder plasticity (lora-moe / top-k), resolution, and resampler
width (which *hurt*). All those encoders are caption/contrastive or general-VLM towers. The one untested class is an
**OCR-native encoder**: GLM-OCR ([`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR)), purpose-built for
dense document OCR — d_enc=1536, patch-14, dynamic high-res (a dense A4 → **~6045 native patches** vs NaFlex's
256-patch cap). This is the **architecture-decision experiment**: it disentangles the two hypotheses we couldn't
separate — *the encoder doesn't supply glyphs* (encoder wall → build a feature-fusion encoder) vs *the frozen
decoder can't read them* (decoder wall → pivot to an agentic OCR-tool).

The decisive design fix (caught mid-experiment): the glyph-reading tasks are the **outcomes**, not controls — they
require the very capability under test. The grounding control must be **glyph-independent**: `vqav2` (sparse-text
semantic VQA), which grounds via general semantics. Decision table: `vqav2>0 + glyph lifts` → encoder wall;
`vqav2>0 + glyph=0` → decoder wall; `vqav2=0` → no grounding (inconclusive).

## Verdict: decoder wall — confirmed

Both runs isolated on the doc family + `vqav2`, NaFlex→GLM-OCR encoder, resampler n_queries=256, pool=1, attn-LoRA
r128. **cheap** = from scratch (no Stage-1); **full** = Stage-1 caption alignment → Stage-2 doc reading
(warm-started), run under autoresearch `SweepRunner` with active GPU triage (0 kills; Stage-1 6300s, Stage-2 9893s).
Each cell **final / peak**:

| task | cheap (scratch) | full (Stage-1→2) | |
|---|---|---|---|
| **vqav2** *(grounding control)* | 0.349 / 0.349 | **0.605 / 0.605** | grounding strongly formed |
| infographic_vqa (semi-dense) | 0.333 / 0.333 | 0.333 / 0.333 | readable, unchanged |
| docvqa | 0.059 / 0.059 | 0.059 / 0.059 | **noise floor** |
| ocrvqa | 0.000 / 0.000 | 0.000 / 0.000 | **dead** |
| visualmrc | 0.000 / 0.143 | 0.000 / 0.000 | **dead** |

**Reading — the encoder is not the wall; the decoder is.**
1. **Grounding demonstrably works, and Stage-1 made it much better.** `vqav2` nearly doubled 0.35→0.61 with proper
   Stage-1 alignment — the projector learned to use GLM-OCR's features for semantic VQA. This rules out the
   "no grounding / inconclusive" branch decisively.
2. **The truly-dense glyph tasks did not move one point.** docvqa 0.059, ocrvqa 0, visualmrc 0 — identical between
   cheap and full, and identical to the SigLIP isolated control (docvqa 0.067, infographic 0.32). An OCR-native
   encoder with **24× the patches** and grounding good enough for vqav2=0.61 bought **exactly nothing** on dense OCR.
3. **Therefore the frozen Laguna decoder cannot transcribe dense glyphs** regardless of how good the visual
   representation is. The native line is exhausted: not the encoder tower, not resolution, not the resampler
   squeeze, not decoder plasticity, not adapter capacity, and now not an OCR-native encoder with proper grounding.

Side-finding: GLM-OCR is also a strong *general* encoder here (vqav2 0.605 > SigLIP-AnyRes/Qwen3-VL's ~0.48–0.64
on their suites) — relevant if a future feature-fusion encoder wants breadth.

## Next move

**Pivot to the agentic / OCR-tool route** — the only avenue that doesn't require the frozen decoder to transcribe
glyphs. Give Laguna an OCR tool in its harness and reward answer correctness (RLVR): the sub-skills exist (reasoning
over extracted text is within the decoder's competence; vqav2=0.61 shows grounding works), so the reward is
non-sparse and RLVR has signal — unlike RLVR on native reading, which sits in a zero-reward desert (docvqa success
rate ≈ 0). Spec: [`docs/specs/2026-06-08-agentic-ocr-tool-design.md`](../../specs/2026-06-08-agentic-ocr-tool-design.md).
The one remaining (parked, expensive, low-prior) native variant is 8-bit-AdamW routed-expert unfreeze — the only
decoder weights never given plasticity.
