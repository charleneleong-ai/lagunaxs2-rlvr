# GRPO / RLVR for the visual adapter — design

## Why
QA-SFT on a frozen base + projector (± attention LoRA) minimizes teacher-forced CE, which is
*not* the metric we care about: multi-turn QA **answer-contains-title** scored **0.000** even as
val CE fell to 3.58. SFT can't optimize a non-differentiable, exact-match target. GRPO can — a
**verifiable reward** (`answer contains the title needle`) is exactly the signal SFT lacks.

## The blocker
TRL `GRPOTrainer` (v1.5.1) drives **both** rollout generation **and** the gradient logprob forward
exclusively through `input_ids` (`_generate_single_turn` → `model.generate(input_ids=...)`;
`_get_per_token_logps_and_entropies` → `model(input_ids=...).logits`). There is **no `inputs_embeds`
entry point**, and the multimodal branch is hard-wired to HF VLM classes that take `pixel_values`.
Our [`VisualAdapter`](../../src/laguna_rlvr/visual/model.py) produces `inputs_embeds` itself
(encoder → projector → splice at the `<image>` marker), so neither stock path fits.

## Decision — wrap the adapter as an `input_ids`-driven HF CausalLM (option b)
Build a thin `PreTrainedModel`-shaped wrapper whose `forward`/`generate` accept `input_ids` carrying
the `<image>` token id, and which splices vision into `inputs_embeds` *internally* before calling the
frozen LLM. This satisfies TRL's generate path **and** its logprob forward with zero trainer surgery;
PEFT (the attention-LoRA + the `"ref"` adapter for the KL term) and reward-kwargs all keep working.
Rejected: (a) subclassing `GRPOTrainer` to override the rollout+logprob internals (they're
`@profiling_decorator` instance methods, overridable, but flagged experimental → upgrade churn);
(c) hand-rolling GRPO (~150 lines, loses TRL's IS/KL/logging — fallback only).

## Components
- **`LagunaCausalLMForGRPO(nn.Module)`** (new, `visual/grpo_model.py`): wraps a `VisualAdapter`.
  - `forward(input_ids, attention_mask=None, labels=None, **kw) -> CausalLMOutputWithPast`: embed
    ids via the frozen embedding, replace each `<image>` position with that row's projected vision
    tokens (reusing [`_embed_with_vision`](../../src/laguna_rlvr/visual/model.py)), call the LLM with
    `inputs_embeds`, return `.logits` (+ `.loss` if labels). Train-forward and generate **must splice
    identically** or rollout logprobs won't match the loss forward → GRPO destabilizes.
  - `generate(input_ids, ...)`: same splice, then `llm.generate(inputs_embeds=...)`.
  - Expose `.config`, `.generation_config`, `gradient_checkpointing_enable`, `save_pretrained`,
    `prepare_inputs_for_generation`, and (PEFT) `add_adapter` — forward to the wrapped LLM.
- **Image threading (out-of-band).** TRL passes only `input_ids` to forward/generate, not our PIL
  images. Dataset carries an `id` column; a module-level `id -> vision_tokens` cache (pre-projected
  once per image) is consulted by the wrapper for the `<image>` splice. Pre-projecting also removes
  the encoder from the hot rollout loop. Variable vision-token counts (variable resolution) expand
  the single `<image>` token into N positions — `attention_mask`/positions must track the expansion
  (TRL left-pads, `padding_side="left"`).
- **Reward** (`reward_funcs`): `def title_match(prompts, completions, title, **kw)` → `1.0` if the
  ground-truth `title` is a substring of the completion (normalized), else `0.0`. `title` arrives as
  a dataset-column kwarg. Same needle logic as [`extract_needle`](../../src/laguna_rlvr/visual/corpora.py).
- **Policy init**: warm-start from the best QA-SFT checkpoint — projector (+ LoRA if the LoRA probe
  helps). GRPO refines the already-engaging policy rather than exploring from scratch.
- **Config**: `GRPOConfig(beta>0)` so the frozen-base KL is a cheap `"ref"` PEFT adapter, not a 2nd
  33B; native transformers generation backend (no vLLM for the custom wrapper); `num_generations` G
  small (4–8) given the 33B rollout cost.

## Open risks
1. **Splice consistency** train vs generate (logprob mismatch) — the top GRPO-specific failure mode.
2. **Variable vision-token count + left-padded batching** — the fiddly correctness surface.
3. **Rollout cost** — G generations/prompt on a 33B; pre-project vision, keep G + max_new_tokens low.

## Gating
Build after the LoRA-SFT probe (`qasftqwenlora`) reports: its result sets the GRPO init (projector
only vs projector+LoRA) but not the GRPO design above. TRL API cross-checked against v1.5.1 docs +
`trl/trainer/grpo_trainer.py`.
