# Visual-context adapter for Laguna XS.2 — giving a blind agent verifiable sight

## Premise

Laguna XS.2 is a **text-only** long-horizon coding agent (Laguna M.1/XS.2 Technical Report, May 2026): no
vision in the architecture. It acts through a terminal harness (`pool`) via GLM-style `<tool_call>` XML,
reasons in `<think>`, and is trained with **verifiable-reward RL (CISPO, §4.4)** over ~1M containerized repos.
Everything it perceives is text.

So a coding agent that can't see can't close the loop on anything visual — the frontend it just wrote, the
chart it generated, the screenshot in the issue, the rendered diff. This adapter is Laguna's **sensory organ**:
a frozen visual encoder → trainable projector → frozen Laguna. The design goal is to add sight as a
**verifiable agentic capability** (see → act → verify by re-render), reusing the report's existing machinery
rather than bolting on passive captioning.

## Architecture

```
image ──▶ frozen encoder ──▶ trainable projector ──▶ N vision tokens
                                                          │  spliced at the <image> marker
prompt/chat-template text  ──▶ embed ──────────────────────┼──▶ inputs_embeds ──▶ frozen Laguna XS.2
```

- **Frozen backbone, BF16.** The report's NVFP4/FP8/INT4 checkpoints store MoE experts as *per-expert
  packed* tensors (§5, QAD) that `compressed-tensors` can't map onto this modeling revision's *fused*
  expert params — they load randomly. Use the unquantized [`poolside/Laguna-XS.2`](https://hf.co/poolside/Laguna-XS.2);
  the load-integrity guard in [`model.py`](../tree/feat/mm-adapter/src/laguna_rlvr/visual/model.py) enforces it.
  Freezing the backbone and training a small delta is also poolside's own quantization pattern.
- **`<image>` placeholder token**, initialized via the report's **new-token recipe (§4.1.1): mean of its
  subtoken embeddings**, then frozen (the projector carries the learning). The token marks *where* projected
  vision tokens are spliced into the sequence — so vision can arrive **anywhere in the chat template**, not
  only as a prefix. This is the foundation for the next point.
- **Vision as a tool observation.** Laguna's world is `tool_call → text observation`. The target integration
  is a `screenshot`/`view_image` tool whose *observation* is the spliced vision tokens — making sight a
  first-class agentic action the model learns to request and consume mid-trajectory (mirrors multi-harness
  SFT §4.3.3 and agentic RL §4.4). The `<image>`-splice mechanism above is what makes this possible.

## Beachhead: UI render-in-the-loop

The agent writes/fixes frontend code → headless-browser screenshot → **sees** it via the adapter → reward =
visual match to target. Highest-value gap for a coding agent, the most naturally verifiable, and it exercises
the full see→act→verify loop. Charts→code is the warm-up; visual bug repair is the same machinery.

## Evaluation: SWE-bench Multimodal (execution-grounded)

[`SWE-bench/SWE-bench_Multimodal`](https://hf.co/datasets/SWE-bench/SWE-bench_Multimodal) — 617 real
GitHub-issue tasks (102 dev / 510 test), mostly JS/frontend, where the `image_assets` field holds the
screenshots/mockups attached to the issue and success is `FAIL_TO_PASS`/`PASS_TO_PASS` tests passing after the
agent's patch. This is the held-out, vision-required, execution-verifiable benchmark — the same shape as the
report's agentic evals (§6.2). It is the bar: a sighted Laguna should beat a blind Laguna here.

**Prime/verifiers status (checked 2026-05):** SWE-bench Multimodal is **not** yet packaged as a Prime
Intellect [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment. The closest Hub entry,
[`vf-multi-swe-bench-openhands`](https://app.primeintellect.ai/dashboard/environments/whyphylabs/vf-multi-swe-bench-openhands),
is **Multi-SWE-bench** (multilingual SWE, not image-issues). `verifiers`/`prime-rl` *do* support multimodal
(VLM) observations, so the plan is to **wrap the HF dataset as a `vf-swe-bench-multimodal` environment** —
modelled on the existing multi-swe-bench env — using the official SWE-bench M execution harness as the binary
verifier and feeding `image_assets` through our adapter. That env is then reusable for both eval and CISPO RL
(stage 3).

## Training ladder (mapped onto the report)

| Stage | What | Report analog |
|---|---|---|
| 0 — baseline | tool-mediated `GLM-OCR → text` (text-dense artifacts only) | the bar the adapter must beat |
| 1 — projector SFT | reconstruction on synthetic OCR (current scaffold, BF16) | imitation / mid-training §4.2 |
| 2 — agentic SFT | Hive-generated trajectories where the agent calls `screenshot` and conditions on vision | §4.3 SFT + §3.2.2 Hive |
| 3 — **agentic RLVR (CISPO)** | render-diff / UI-match / test-pass as the binary verifier reward | §4.4 — the payoff |

**Reward stays verifiable.** §4.4.2's deterministic checker chain (binary task verifier = repo tests / shell
assertions) extends directly: render the agent's output → image → match to target / tests pass. A render-diff
is a deterministic verifier, so it drops into CISPO with no reward model.

## Data: self-labeling, verifiable

Reuse the report's "real git commit → verifiable task" trick (§4.3.1) for *visual* diffs:
- Render real frontend / matplotlib sources → `(screenshot, source)` pairs (infinite, self-labeling).
- Mutation tasks: take a working UI, inject a visual bug (CSS) → target is the screenshot of the correct
  version → verify by re-render diff.
- SWE-bench Multimodal `image_assets` + test specs as the real-world held-out set.

## Guardrails (already in place)

- A100 config gate — `mm_adapter_plan.py` must print `A100-40GB guardrails: pass` before any heavyweight run.
- Load-integrity guard — fails loudly if any backbone weight loads random (the NVFP4 trap).
- Determinism — `seed_everything()` + `--seed` (default 42).
- Always-logged `results.jsonl` via `GPUMonitor` (autoresearch).

## Non-goals (this stage)

- Training Laguna's weights (backbone stays frozen; QLoRA only if projector-only saturates).
- Modifying MoE routing, tokenizer internals beyond the `<image>` token, or attention architecture.
- Generic image understanding — the target is code/agentic visual artifacts, not captioning.
