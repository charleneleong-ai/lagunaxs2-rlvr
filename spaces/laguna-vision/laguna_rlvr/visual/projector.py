from __future__ import annotations

import torch
from torch import nn


def mean_pool(x: torch.Tensor, k: int) -> torch.Tensor:
    """Average every k adjacent tokens: (B, N, D) -> (B, N//k, D). k=1 is identity.

    Drops a tail of N % k tokens so N is divisible by k.
    """
    if k <= 1:
        return x
    b, n, d = x.shape
    n_keep = (n // k) * k
    return x[:, :n_keep, :].reshape(b, n_keep // k, k, d).mean(dim=2)


class Projector(nn.Module):
    """Maps frozen-encoder features (d_in) into the LLM embedding space (d_out).

    The only trainable module in the adapter. `linear` = LLaVA-1.0; `mlp` = LLaVA-1.5.
    """

    def __init__(self, d_in: int, d_out: int, kind: str = "linear"):
        super().__init__()
        if kind == "linear":
            self.net: nn.Module = nn.Linear(d_in, d_out)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        else:
            raise ValueError(f"unknown projector kind {kind!r} (use 'linear' or 'mlp')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
