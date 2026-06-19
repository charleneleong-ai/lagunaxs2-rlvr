"""Encoder-free VLM POC launcher: smoke (Qwen3-0.6B) then ship (Laguna-XS.2), sequential one-GPU.

Reuses the standard visual projector-SFT recipe (same dataset + eval as the SigLIP baseline) and only
swaps in the encoder-free path: `--encoder patchify --projector patch_embed --pool 1`. `--no-anchor`
because the PatchEmbedder's own LayerNorms set the token scale (the soft anchor would fight them).

The smoke validates the embedder learns to *see* on a small decoder (per the reference's "scale the
decoder" finding — SmolLM2-135M was too weak); the ship run is the real target. Each is a blocking
`train.py` subprocess, so they serialize naturally on the single GPU.

    uv run python scripts/encoder_free_poc.py                 # smoke 400 steps, ship 800
    uv run python scripts/encoder_free_poc.py --steps-smoke 50 --steps-ship 0   # ship 0 = skip
"""
from __future__ import annotations

import subprocess
import sys

import typer

app = typer.Typer(add_completion=False)

_EF_FLAGS = ["--encoder", "patchify", "--projector", "patch_embed", "--pool", "1", "--no-anchor",
             "--no-wandb"]


def _train(base: str, steps: int, suffix: str) -> None:
    print(f"\n=== TRAIN {suffix} ({base}, {steps} steps) ===", flush=True)
    subprocess.run([sys.executable, "-m", "laguna_rlvr.visual.train", *_EF_FLAGS,
                    "--base", base, "--steps", str(steps), "--name-suffix", suffix], check=False)


@app.command()
def main(steps_smoke: int = typer.Option(400, help="Qwen3-0.6B smoke steps (0 = skip)"),
         steps_ship: int = typer.Option(800, help="Laguna-XS.2 ship steps (0 = skip)")) -> None:
    if steps_smoke:
        _train("Qwen/Qwen3-0.6B", steps_smoke, "ef_smoke")
    if steps_ship:
        _train("poolside/Laguna-XS.2", steps_ship, "ef_ship")


if __name__ == "__main__":
    app()
