"""External-benchmark suite — run the standard public benchmarks for each capability the adapter is
meant to add, through one registry. Each entry loads a fixed held-out set and scores it with that
benchmark's protocol; the scorers return `{name/metrics/...}` so they log straight into the panel
alongside the in-house qa/multi-turn metrics.

Capability coverage (benchmark -> what it proves the adapter can do):
  read        OCRBench        — perceive/transcribe text in the image (the confabulation bottleneck)
  vqa         MMMU, MathVista — reason over the image, not just read it
  visual-code Design2Code     — turn a screenshot into faithful code (the coding-model thesis)
  ground      ScreenSpot-v2   — localize a UI element by instruction (the agentic/act half)
  converse    MMDU            — hold a multi-turn multi-image dialogue (long-horizon memory)

Deferred to a next stage (need heavier infra — see docs/specs/external-benchmarks.md):
  visual-code-by-execution  ChartMimic / Plot2Code  (chart->code, run + render-diff: code-exec sandbox)
  agentic task envs         VisualWebArena / OSWorld / Multimodal-Mind2Web  (full env rollout harness;
                            the in-house frontend_design / ocr_tool verifiers envs already cover tool-use)
"""
from __future__ import annotations

import functools

import torch
import typer

from laguna_rlvr.visual import design2code, grounding, mmdu, mmmu, ocrbench, screenspot
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import VisualAdapter

# name -> (load items for `n` examples, score them). Scorers all take (adapter, items) -> {k: float};
# screenspot reuses the shared grounding scorer with its own prefix.
BENCHMARKS = {
    "ocrbench": lambda n: (ocrbench.OCRBenchDataset(n=n), ocrbench.ocrbench_eval),
    "mmmu": lambda n: (mmmu.MMMUDataset(n=n), mmmu.mmmu_eval),
    "mathvista": lambda n: (mmmu.MathVistaDataset(n=n), mmmu.mathvista_eval),
    "design2code": lambda n: (design2code.Design2Code(n=n), design2code.design2code_eval),
    "screenspot_v2": lambda n: (screenspot.ScreenSpotV2Dataset(n=n),
                                functools.partial(grounding.screenspot_eval, prefix="screenspot")),
    "mmdu": lambda n: (mmdu.MMDUDataset(n=n), mmdu.mmdu_eval),
}
DEFAULT_SUITE = list(BENCHMARKS)


def run_benchmarks(adapter: VisualAdapter, names: list[str], n: int = 64,
                   run=None, step: int | None = None) -> dict[str, float]:
    """Run each named benchmark on `n` held-out examples; print + (optionally) W&B-log its metrics."""
    if unknown := [name for name in names if name not in BENCHMARKS]:
        raise ValueError(f"unknown benchmark(s) {unknown} (have: {', '.join(BENCHMARKS)})")
    results: dict[str, float] = {}
    for name in names:
        dataset, scorer = BENCHMARKS[name](n)
        items = [dataset[i] for i in range(len(dataset))]
        metrics = scorer(adapter, items)
        results.update(metrics)
        print(f"[{name}] " + "  ".join(f"{k.split('/')[-1]} {v:.3f}" for k, v in metrics.items()), flush=True)
        if run is not None:
            run.log(metrics, step=step)
    return results


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    ckpt: str = typer.Option(..., help="adapter checkpoint (projector + LoRA) to evaluate"),
    encoder: str = "siglip", base: str = "poolside/Laguna-XS.2", projector: str = "resampler",
    suite: str = typer.Option(",".join(DEFAULT_SUITE), help="comma-list of benchmarks to run"),
    n: int = typer.Option(64, help="held-out examples per benchmark"),
) -> None:
    """Evaluate a trained adapter on the external benchmark suite (the metric IS the capability)."""
    adapter = VisualAdapter(load_encoder(encoder, pool=(4 if "qwen" in encoder else 1)), base,
                            projector_kind=projector, use_anchor=False)
    adapter.load_adapter_state_dict(torch.load(ckpt, map_location=adapter.llm.device))
    run_benchmarks(adapter, [s for s in suite.split(",") if s], n=n)


if __name__ == "__main__":
    app()
