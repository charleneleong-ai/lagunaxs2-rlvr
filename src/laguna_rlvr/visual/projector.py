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


class _ResamplerBlock(nn.Module):
    """One pre-LN cross-attention + FFN block: queries attend to the encoder features (kv)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_ff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kvn = self.ln_kv(kv)
        q = q + self.attn(self.ln_q(q), kvn, kvn, need_weights=False)[0]
        return q + self.ff(self.ln_ff(q))


class Resampler(nn.Module):
    """Perceiver-style resampler: `n_queries` learned queries cross-attend to the (variable-length)
    encoder features and emit a *fixed* `n_queries` tokens in the LLM space. A Perceiver/Q-Former-style
    connector — it both bridges d_in->d_out and compresses AnyRes tiles (global + hi-res crops,
    thousands of patches) to a constant token budget, sidestepping variable-length splicing.
    """

    def __init__(self, d_in: int, d_out: int, n_queries: int = 256, n_heads: int = 8, depth: int = 2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_out) * 0.02)
        self.kv = nn.Linear(d_in, d_out)
        self.blocks = nn.ModuleList(_ResamplerBlock(d_out, n_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d_out)
        # learnable output projection AFTER the norm: a LayerNorm alone PINS the output norm
        # ~sqrt(d_out), forcing reliance on the soft anchor; a trailing Linear lets the resampler
        # learn its own output scale, so --no-anchor is viable.
        self.out = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_in) -> (B, n_queries, d_out)
        kv = self.kv(x)
        q = self.query.to(x.dtype).unsqueeze(0).expand(x.size(0), -1, -1)
        for block in self.blocks:
            q = block(q, kv)
        return self.out(self.norm(q))


class Projector(nn.Module):
    """Maps frozen-encoder features (d_in) into the LLM embedding space (d_out).

    The only trainable module in the adapter. `linear` = LLaVA-1.0; `mlp` = LLaVA-1.5;
    `resampler` = Perceiver connector emitting a fixed token count.
    """

    def __init__(self, d_in: int, d_out: int, kind: str = "linear", n_queries: int = 256):
        super().__init__()
        if kind == "linear":
            self.net: nn.Module = nn.Linear(d_in, d_out)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        elif kind == "resampler":
            self.net = Resampler(d_in, d_out, n_queries=n_queries)
        else:
            raise ValueError(f"unknown projector kind {kind!r} (use 'linear', 'mlp', or 'resampler')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_compatible(self, sd: dict) -> list[str]:
        """Warm-start from `sd` (a wrapped checkpoint or a raw projector state_dict), loading only params
        whose shape matches the current module and leaving the rest at init — e.g. a resized resampler
        query bank when `n_queries` changed (the cross-attn / kv / FFN weights are shape-independent of
        query count and input length, so the grounding machinery transfers and only the new queries
        relearn). Returns the keys left at init."""
        sd = sd["projector"] if "projector" in sd else sd  # accept the checkpoint envelope or a raw sd
        own = self.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        skipped = [k for k in own if k not in keep]
        self.load_state_dict(keep, strict=False)
        return skipped
