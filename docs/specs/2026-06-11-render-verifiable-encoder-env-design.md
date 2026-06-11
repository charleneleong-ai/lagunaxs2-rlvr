# Render-verifiable encoder env — chart/screenshot → code, scored by re-rendering

> Scoping doc. The encoder+decoder+tool portfolio member with a **deterministic render reward** — the one
> kind of multimodal reward RLVR most wants (no reward model, no answer-string brittleness).

## Why

`VisionToolEnv` (built) scores the encoder+decoder+tool loop by *answer-match* on VQA. That's brittle on
generative tasks (the chart2text/ROUGE-L lesson). The strongest multimodal RLVR signal is **render-diff**:
the model sees an image (encoder), emits the *code that reproduces it*, and the reward is how close the
**re-rendered** output is to the original. Two benchmarks already in the repo give exactly this:

| benchmark | input → output | re-render | reward primitive | weight |
|---|---|---|---|---|
| **ChartMimic** (`corpora._chartmimic`) | chart PNG → matplotlib code (`GroundTruthFigureCode`) | `python` exec → PNG | image-similarity vs gold figure | **start here** |
| **Design2Code** (`visual/design2code.py`) | screenshot → HTML/CSS | headless browser → PNG | render-diff (held-out ranker) | phase 2 |

**Start with ChartMimic**: matplotlib re-render is a `subprocess` away (the existing `code_exec` path),
whereas Design2Code needs a headless-browser sandbox. Same env shape, lighter blocker.

## Shape

Reuse `VisionToolEnv`'s local-adapter loop ([`vision_tool_eval.py`](../../src/laguna_rlvr/visual/vision_tool_eval.py)) —
the encoder channel (image spliced) is identical. The differences:
- **Output is code, not an answer.** The terminal action emits a fenced code block; no `ocr` tool (there's
  no glyph text to read — the signal is structural, the encoder's job).
- **Reward is render-diff, not `_match`.** A new `score_render(code, gold_image)`:
  1. exec the model's matplotlib code in a `subprocess` (timeout, like [`code_exec.score_code`](../../src/laguna_rlvr/code_exec.py)) → PNG
  2. image-similarity (SSIM or a perceptual/embedding distance) between the re-rendered PNG and the gold figure
  3. blend with `code_valid` (did it exec at all) — a `vf.Rubric`, mirroring `agentic_repair`'s composite

## Reuse vs build

| piece | exists | source |
|---|---|---|
| chart image + gold code | ✅ | `corpora._chartmimic` (`image_col=GroundTruthFigurePreview`, `text_col=GroundTruthFigureCode`) |
| encoder+decoder loop | ✅ | `vision_tool_eval.run_episode` (drop the ocr tool; terminal = code block) |
| subprocess code exec + timeout | ✅ | `code_exec.score_code` (generalize: capture a written PNG, not assert pass/count) |
| CodeBLEU (secondary signal) | ✅ | `visual/code_metrics.py` |
| composite reward rubric | ✅ | `rewards.py` + `agentic_repair`'s pattern |
| **image-similarity reward** | ❌ | **build** — `score_render`: SSIM/perceptual diff (the only new reward primitive) |
| **matplotlib render sandbox** | ⚠️ | subprocess + timeout is the MVP (matplotlib `Agg`, no display); container only if untrusted |

## Blocker

The render sandbox. ChartMimic keeps it small: matplotlib code is re-rendered headless (`matplotlib.use("Agg")`)
in a timed subprocess — no Docker, no browser. The doc's Stage-3 "code-execution sandbox" worry
([a100-multimodal-adapter.md](../a100-multimodal-adapter.md)) is real for SWE-bench M (arbitrary repos), but
**not** for ChartMimic (self-contained plotting). So this is the cheapest path to a verifiable multimodal
reward — and a stepping stone to Design2Code (swap the renderer for a headless browser) and SWE-bench M
(swap render-diff for test-execution).

## Sequence

1. `score_render(code, gold_png) -> (valid, similarity)` — the one new primitive (+ tests on a known
   matplotlib snippet vs its render).
2. `RenderToolEnv` (or a `--task render` mode on `vision_tool_eval`) — encoder loop, code terminal,
   `score_render` reward → `results/probe/render_chartmimic__<slug>.jsonl`, ranked like the rest.
3. Probe ChartMimic for learnable signal; if `signal > 0`, it joins the encoder+decoder+tool training set.
4. Phase 2: Design2Code (headless browser). Phase 3: SWE-bench M (test-execution) — gap #2.
