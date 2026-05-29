# A100 40GB generalized adapter scaffold
This scaffold keeps `feat/laguna-rlvr` focused on generalized learning while avoiding a risky full architecture fork. The first native-adapter stage is projector-only:
- freeze Laguna XS.2, preferably the NVFP4/4-bit checkpoint
- freeze the modality encoder, starting with SigLIP for images
- train a small resampler/projector that maps modality embeddings into a short sequence of learned tokens
- prepend or interleave those learned tokens with the existing text prompt
- postpone QLoRA and RLVR until projector-only supervised alignment shows a nonzero signal
## Why this fits the branch
The existing RLVR stack already gates training on learnable reward variance. This adapter scaffold adds an earlier gate: a config must be safe for one A100 40GB before any heavyweight training code runs. It also leaves room for image, audio, UI-state, or tool-observation encoders behind the same adapter interface.
## Dry run
```bash
uv run python scripts/mm_adapter_plan.py configs/mm_adapter/a100-40gb-projector.toml
```
Expected result: the config prints the frozen backbone, frozen encoder, projector settings, effective batch size, and `A100-40GB guardrails: pass`.
## Staging path
1. Projector SFT on tiny image-text or screenshot-text alignment data.
2. Add optional QLoRA only if projector-only alignment is too weak.
3. Convert aligned examples into verifiable tasks, such as screenshot-to-code or diagram-to-test cases.
4. Run the existing probe/report loop and only launch RLVR when reward variance is nonzero.
## Non-goals
- full BF16 Laguna training on one A100 40GB
- long-context multimodal agent trajectories in the first stage
- modifying MoE routing, tokenizer internals, or attention architecture
