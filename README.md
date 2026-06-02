# laguna-rlvr

Finetuning [`poolside/Laguna-XS.2`](https://hf.co/poolside/Laguna-XS.2) (a 33B-MoE text-only coding agent) along two tracks: a **self-evolving text RLVR loop** on Prime Intellect, and a **vision adapter** that gives the blind agent verifiable sight.

## Two tracks

**1. Text RLVR (self-evolving).** Synthesize verifiable tasks, probe the base model's pass rate, rank by *learnable signal*, then gate/train via verifiable-reward RL on Prime Intellect. Core in [`src/laguna_rlvr/`](src/laguna_rlvr/) (`synth.py`, `probe.py`, `report.py`, `rl.py`, `rewards.py`), with `verifiers`-style environments under [`environments/`](environments/) and RL configs in [`configs/rl/`](configs/rl/).

**2. Vision adapter.** A frozen vision encoder (SigLIP2-NaFlex / AnyRes, or Qwen3-VL) -> trainable Perceiver resampler (256 vision tokens, +attention LoRA) -> frozen Laguna, with vision spliced at an `<image>` marker so it arrives as a tool observation, not a fixed prefix. Trained by QA-SFT (forces vision use) then verifiable-reward RL (GSPO). Code in [`src/laguna_rlvr/visual/`](src/laguna_rlvr/visual/); full design in [`docs/a100-multimodal-adapter.md`](docs/a100-multimodal-adapter.md).

## Quickstart

```bash
mise run setup     # install deps + the Prime Intellect CLI
mise run doctor    # verify the toolchain
mise run test      # run the test suite
```

Common tasks (run `mise tasks` for the full list):

| Task | What |
|---|---|
| `mise run probe -- env=<env> model=<model>` | probe a model's pass rate on an environment |
| `mise run report` | rank probe results by learnable signal |
| `mise run rlvr-loop -- <domains>` | one self-evolving turn: synthesize -> probe -> rank -> gate/train |
| `mise run train` | launch the hosted Laguna RL run from the toml config |
| `mise run mm-adapter-plan` | dry-run the vision-adapter VRAM-budget gate |

Vision-adapter training (SFT / RL) is driven directly via `python -m laguna_rlvr.visual.train` and `...visual.gspo` (see the design doc for the recipe and flags).

## Layout

```
src/laguna_rlvr/        core: probe / synth / report / rl / rewards / mm_adapter
src/laguna_rlvr/visual/ vision adapter: encoders, projector, model, train, gspo, eval
environments/           verifiers-style RL environments (swe_multilingual, ocr_tool, ...)
configs/ · conf/        Hydra configs + RL / mm-adapter configs
docs/                   design specs + the multimodal-adapter architecture doc
mise.toml               task runner entrypoints
```

## Docs

- [`docs/a100-multimodal-adapter.md`](docs/a100-multimodal-adapter.md) - vision-adapter architecture, training ladder, data, metrics
- [`docs/specs/`](docs/specs/) - design specs: finetune structure, Stage-0 baseline panel, GRPO/RLVR
