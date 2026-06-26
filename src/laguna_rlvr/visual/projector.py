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


class PatchEmbedder(nn.Module):
    """Encoder-free 'embedder' (Gemma 4 / Fuyu style): raw flattened pixel patches -> LLM tokens, with
    NO pretrained vision tower. Input is (B, N, patch_dim) raw-pixel patches from `PatchifyEncoder`;
    output is (B, N, d_out) ready to splice at <image>. The whole module is trainable — it IS the bridge.

    Recipe (matches Gemma 4 12B's embedder): LayerNorm -> Linear(patch_dim -> d_out) -> LayerNorm ->
    add factorized row/col positional embeddings -> LayerNorm -> connector Linear(d_out -> d_out). The
    factorized table (`E_row[i] + E_col[j]`) costs 2·g·d_out params vs g²·d_out for a full g×g table.

    The grid side is read from the incoming token count at forward (`g = √N`), so the same module
    serves any square patch grid up to `max_grid` — no per-encoder grid wiring.
    """

    def __init__(self, patch_dim: int, d_out: int, max_grid: int = 64):
        super().__init__()
        self.ln_in = nn.LayerNorm(patch_dim)
        self.fc = nn.Linear(patch_dim, d_out)
        self.ln_proj = nn.LayerNorm(d_out)
        self.row_emb = nn.Parameter(torch.randn(max_grid, d_out) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(max_grid, d_out) * 0.02)
        self.ln_pos = nn.LayerNorm(d_out)
        self.connector = nn.Linear(d_out, d_out)

    def _positions(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        """The `n` factorized positions in row-major order: pos(i, j) = E_row[i] + E_col[j], grid g = √n."""
        g = round(n ** 0.5)
        if g * g != n:
            raise ValueError(f"patch_embed expects a square patch grid; got {n} tokens (√n not integer)")
        pos = self.row_emb[:g].unsqueeze(1) + self.col_emb[:g].unsqueeze(0)  # (g, g, d)
        return pos.reshape(n, -1).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, patch_dim) -> (B, N, d_out)
        x = self.ln_proj(self.fc(self.ln_in(x)))
        x = self.ln_pos(x + self._positions(x.shape[1], x.dtype).unsqueeze(0))
        return self.connector(x)


class Projector(nn.Module):
    """Maps frozen-encoder features (d_in) into the LLM embedding space (d_out).

    The only trainable module in the adapter. `linear` = LLaVA-1.0; `mlp` = LLaVA-1.5;
    `resampler` = Perceiver connector emitting a fixed token count; `patch_embed` = encoder-free
    Gemma-4/Fuyu embedder mapping raw pixel patches straight in (see `PatchEmbedder`).
    """

    def __init__(self, d_in: int, d_out: int, kind: str = "linear", n_queries: int = 256):
        super().__init__()
        if kind == "linear":
            self.net: nn.Module = nn.Linear(d_in, d_out)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        elif kind == "resampler":
            self.net = Resampler(d_in, d_out, n_queries=n_queries)
        elif kind == "patch_embed":
            self.net = PatchEmbedder(d_in, d_out)
        else:
            raise ValueError(
                f"unknown projector kind {kind!r} (use 'linear', 'mlp', 'resampler', or 'patch_embed')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_compatible(self, sd: dict) -> list[str]:
        """Warm-start from `sd` (a wrapped checkpoint or a raw projector state_dict). Exact-shape params
        load whole; a shape-grown param (e.g. a resampler query bank resized via `n_queries`) keeps its
        overlapping leading sub-block from `sd` and leaves the remainder at init — so a 256→1024 query
        bank reproduces the trained 256 queries' grounding and only the extra 768 start fresh, instead of
        discarding the Stage-1 alignment wholesale. The cross-attn / kv / FFN weights are shape-independent
        of query count and input length, so they transfer whole. Returns the partially-loaded keys."""
        sd = sd["projector"] if "projector" in sd else sd  # accept the checkpoint envelope or a raw sd
        partial = []
        with torch.no_grad():
            for k, dst in self.state_dict().items():
                if k not in sd:
                    continue
                src = sd[k].to(dst.device, dst.dtype)
                if src.shape == dst.shape:
                    dst.copy_(src)
                else:  # grow/shrink along any dim: copy the overlapping leading block, keep the rest at init
                    sl = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
                    dst[sl].copy_(src[sl])
                    partial.append(k)
        return partial
