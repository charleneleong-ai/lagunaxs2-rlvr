# External benchmark suite

Standard public benchmarks wired into the eval panel, one per capability the visual adapter is meant
to add. The scorers live in `src/laguna_rlvr/visual/`, registered in
[`benchmarks.py`](../../src/laguna_rlvr/visual/benchmarks.py) (`BENCHMARKS`), and run either standalone
(`python -m laguna_rlvr.visual.benchmarks --ckpt <best.pt> --suite ...`) or in `train.py`'s final probe
(`--benchmarks ocrbench,mmmu,... --bench-n 64`).

## Capability map — what each benchmark adds vs. what the task requires

The thesis: graft vision onto a strong **text coding model** for **agentic coding** use. That requires
five capabilities; each wired benchmark is the standard external check for exactly one, so the panel
reads as a capability dashboard rather than a single score.

| Required capability | Benchmark (now) | Metric | Why it's the bottleneck |
| --- | --- | --- | --- |
| **Read** — perceive/transcribe image text | [OCRBench](../../src/laguna_rlvr/visual/ocrbench.py) | substring acc (+per-category) | The confabulation wall: the model invents text instead of reading it. |
| **Reason** — answer over the image | [MMMU](../../src/laguna_rlvr/visual/mmmu.py), MathVista | multiple-choice / numeric acc | Reading ≠ reasoning; MMMU/MathVista need inference over the pixels. |
| **Visual-code** — screenshot → faithful code | [Design2Code](../../src/laguna_rlvr/visual/design2code.py) | rendered visual + block/text/position/color | The headline capability — vision serving the coding model. |
| **Ground** — localize a UI element | [ScreenSpot-v2](../../src/laguna_rlvr/visual/screenspot.py) | IoU@0.5 / center-acc | The agentic *act* half: point at the thing to click. |
| **Converse** — multi-turn multi-image dialogue | [MMDU](../../src/laguna_rlvr/visual/mmdu.py) | per-turn overlap + cross-turn recall | Long-horizon use needs memory across turns, not one-shot. |

Two scoring caveats, by design (documented in-module):
- **Design2Code visual similarity** reuses the adapter's own vision encoder (cosine of rendered
  screenshots) instead of a separate CLIP download — same "do the pages look alike" signal, no extra dep.
- **MMDU** uses a lightweight reference-overlap proxy, not the official GPT-4 multi-dimensional judge —
  enough for in-loop tracking; swap in a judge for a headline number.

## Deferred to next stage

Saved deliberately — both need infrastructure beyond a metric:

- **Visual-code by execution — ChartMimic / Plot2Code** (chart image → matplotlib code). The faithful
  metric runs the generated code, renders it, and diffs against the target chart — so it needs the
  **code-execution sandbox**. Design2Code (web→HTML, render via Playwright) covers the screenshot→markup
  axis now; the chart→code axis lands when the sandbox is wired.
- **Agentic task envs — VisualWebArena / OSWorld / Multimodal-Mind2Web.** These are full browser/OS
  **rollout environments** (task-success over long horizons), a whole harness rather than a scorer. The
  in-house `frontend_design` / `ocr_tool` verifiers envs already give us tool-use long-horizon coverage,
  so the external agentic suites are a separate effort, not part of the metrics panel.
