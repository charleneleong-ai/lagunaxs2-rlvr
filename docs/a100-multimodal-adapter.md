# A100 40GB visual-context adapter scaffold
This scaffold keeps `feat/laguna-rlvr` focused on generalized learning while avoiding a risky full architecture fork. The first native-adapter stage is GLM-OCR-first and projector-only:
- freeze Laguna XS.2, preferably the NVFP4/4-bit checkpoint
- use GLM-OCR as the primary OCR/context encoder for document, code, and screenshot images
- keep SigLIP2 NaFlex as a simpler fallback encoder for generic visual features
- train a small context projector that maps OCR/document features into a short sequence of learned tokens
- prepend or interleave those learned tokens with the existing text prompt
- postpone QLoRA and RLVR until projector-only supervised alignment shows a nonzero signal
## Why this fits the branch
The existing RLVR stack already gates training on learnable reward variance. This adapter scaffold adds an earlier gate: a config must be safe for one A100 40GB before any heavyweight training code runs. GLM-OCR is the better first encoder for Laguna because the target is visual text/code/document context compression, not generic image understanding.
## Dry run
```bash
uv run python scripts/mm_adapter_plan.py configs/mm_adapter/a100-40gb-projector.toml
```
Expected result: the config prints the frozen backbone, frozen encoder, projector settings, effective batch size, and `A100-40GB guardrails: pass`.
## Staging path
1. Tool-mediated GLM-OCR path: image/PDF/screenshot → Markdown or JSON → Laguna text/code agent → verifier reward.
2. Projector SFT on tiny OCR reconstruction examples: rendered code/docs/screenshots → original text or code.
3. Add optional QLoRA only if projector-only alignment is too weak.
4. Convert aligned examples into verifiable tasks, such as screenshot-to-code, visual bug repair, or document-to-test cases.
5. Run the existing probe/report loop and only launch RLVR when reward variance is nonzero.
## Non-goals
- full BF16 Laguna training on one A100 40GB
- long-context multimodal agent trajectories in the first stage
- modifying MoE routing, tokenizer internals, or attention architecture
