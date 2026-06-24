from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import (AutoImageProcessor, AutoModelForImageTextToText, Siglip2VisionModel,
                          SiglipVisionModel)

from laguna_rlvr.visual.projector import mean_pool

_REPOS = {"glm_ocr": "zai-org/GLM-OCR", "qwen3_vl": "Qwen/Qwen3-VL-4B-Instruct",
          "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct"}  # stronger general vision tower for the projector
# SigLIP 2 so400m: retrained (vs SigLIP) with caption + self-distill + dense losses -> better
# localized features (OCR/UI). 384 = fixed-square + AnyRes tiling; naflex = native aspect-ratio /
# variable resolution (no tiling -> the encoder ingests the page at its own shape).
_SIGLIP_REPO = "google/siglip2-so400m-patch16-384"
_SIGLIP_NAFLEX_REPO = "google/siglip2-so400m-patch16-naflex"


def _anyres_tiles(img, size: int = 384, grid: int = 2) -> list:
    """AnyRes views: one global thumbnail + a grid*grid set of higher-detail crops. The crops give
    the encoder the resolution to actually *resolve* small text (titles/OCR) — a single down-scaled
    view can't. Returns 1 + grid*grid same-size tiles."""
    img = img.convert("RGB")
    tiles = [img.resize((size, size))]
    big = img.resize((size * grid, size * grid))
    for i in range(grid):
        for j in range(grid):
            tiles.append(big.crop((j * size, i * size, (j + 1) * size, (i + 1) * size)))
    return tiles

# Both GlmOcrForConditionalGeneration and Qwen3VLForConditionalGeneration nest the
# frozen vision tower at `.model.visual`. Its forward is `visual(pixel_values, grid_thw=...)`
# and returns a BaseModelOutputWithPooling whose `.pooler_output` holds the per-patch
# features (already spatial-merged to `out_hidden_size`) that get spliced into the LLM.
_TOWER_PATH = ("model", "visual")


def _resolve(module: torch.nn.Module, path: tuple[str, ...]) -> torch.nn.Module:
    for attr in path:
        module = getattr(module, attr)
    return module


@dataclass
class Encoder:
    tower: torch.nn.Module
    processor: object
    d_enc: int
    pool: int

    @torch.no_grad()
    def encode(self, images: list) -> torch.Tensor:
        """Frozen patch features for `images` -> (B, N/pool, d_enc).

        Each image yields a variable number of patches (depends on its resolution), so
        images are encoded one at a time and stacked. With same-size inputs (the common
        case for synthetic OCR) this gives a clean (B, N/pool, d_enc) batch; truly
        variable-sized inputs would need padding, which we leave to the caller.
        """
        device = next(self.tower.parameters()).device
        dtype = next(self.tower.parameters()).dtype
        feats = []
        for img in images:
            batch = self.processor(images=[img], return_tensors="pt").to(device)
            out = self.tower(
                batch["pixel_values"].to(dtype),
                grid_thw=batch["image_grid_thw"],
            )
            feats.append(out.pooler_output)  # (N, d_enc)
        x = torch.stack(feats, dim=0)  # (B, N, d_enc)
        return mean_pool(x, self.pool)


@dataclass
class SiglipAnyResEncoder:
    """SigLIP vision tower over AnyRes tiles. Same interface as `Encoder` (encode -> (B, N, d_enc),
    d_enc, pool) so it drops into VisualAdapter unchanged. Each image -> (1+grid^2) tiles ->
    fixed N = tiles * patches per image, meant to feed a `resampler` projector that compresses to a
    constant token budget."""
    tower: torch.nn.Module
    processor: object
    d_enc: int
    pool: int = 1
    grid: int = 2

    @torch.no_grad()
    def encode(self, images: list) -> torch.Tensor:
        device = next(self.tower.parameters()).device
        dtype = next(self.tower.parameters()).dtype
        feats = []
        for img in images:
            tiles = _anyres_tiles(img, grid=self.grid)
            px = self.processor(images=tiles, return_tensors="pt")["pixel_values"].to(device, dtype)
            out = self.tower(px).last_hidden_state  # (tiles, patches, d_enc)
            feats.append(out.reshape(-1, out.shape[-1]))  # (tiles*patches, d_enc)
        return mean_pool(torch.stack(feats, dim=0), self.pool)  # (B, tiles*patches, d_enc)


@dataclass
class Siglip2NaflexEncoder:
    """SigLIP2-NaFlex tower: native aspect-ratio / variable-resolution patches (no AnyRes tiling). The
    processor emits padded `pixel_values` + a `pixel_attention_mask` + `spatial_shapes`; we drop the
    padded patches so the resampler sees only real ones. Same (encode -> (B, N, d_enc)) interface."""
    tower: torch.nn.Module
    processor: object
    d_enc: int
    pool: int = 1

    @torch.no_grad()
    def encode(self, images: list) -> torch.Tensor:
        device = next(self.tower.parameters()).device
        dtype = next(self.tower.parameters()).dtype
        feats = []
        for img in images:
            b = self.processor(images=[img.convert("RGB")], return_tensors="pt").to(device)
            out = self.tower(pixel_values=b["pixel_values"].to(dtype),
                             pixel_attention_mask=b["pixel_attention_mask"], spatial_shapes=b["spatial_shapes"])
            feats.append(out.last_hidden_state[0][b["pixel_attention_mask"][0].bool()])  # drop padded patches
        # variable N per image -> only stackable at micro_batch=1 (our setting); the resampler folds N away.
        return mean_pool(torch.stack(feats, dim=0), self.pool)


def extract_flattened_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, nh·nw, C·P·P): split into P×P patches, flatten each in row-major order.
    A single reshape→permute→reshape (no Python loop) so it stays one parallel GPU op."""
    b, c, h, w = x.shape
    p, nh, nw = patch_size, h // patch_size, w // patch_size
    x = x.reshape(b, c, nh, p, nw, p).permute(0, 2, 4, 1, 3, 5)  # (B, nh, nw, C, P, P)
    return x.reshape(b, nh * nw, c * p * p)


@dataclass
class PatchifyEncoder:
    """Encoder-free patchifier (Gemma 4 / Fuyu style): NO pretrained tower. Standardizes each image
    (resize shorter side -> `img_size`, center-crop square) and emits raw flattened pixel patches
    (B, grid², 3·P·P) in [0, 1]. `d_enc = 3·P·P`; the trainable `PatchEmbedder` projector turns these
    into LLM tokens. Same `encode -> (B, N, d_enc)` interface as `Encoder`, so it drops into
    VisualAdapter unchanged — the heavy frozen tower is replaced by a resize + reshape."""

    img_size: int = 512
    patch_size: int = 32
    pool: int = 1

    def __post_init__(self):
        # Encoder-free is strictly 1:1 patch->token; pooling raw patches scrambles the row-major layout
        # the patch_embed positional table assumes. Reject it loudly rather than emit misaligned tokens.
        if self.pool != 1:
            raise ValueError(f"patchify is 1:1 patch->token; pool must be 1, got {self.pool}")

    @property
    def d_enc(self) -> int:
        return 3 * self.patch_size ** 2

    @property
    def grid(self) -> int:
        return self.img_size // self.patch_size

    def _standardize(self, img) -> torch.Tensor:
        """Resize shorter side to `img_size` (upscales small images), center-crop, -> (3, S, S) in [0, 1]."""
        img = img.convert("RGB")
        w, h = img.size
        scale = self.img_size / min(w, h)
        img = img.resize((max(round(w * scale), self.img_size), max(round(h * scale), self.img_size)))
        w, h = img.size
        left, top = (w - self.img_size) // 2, (h - self.img_size) // 2
        img = img.crop((left, top, left + self.img_size, top + self.img_size))
        arr = torch.from_numpy(np.array(img, dtype=np.uint8))  # (S, S, 3); np.array copies -> writable
        return arr.permute(2, 0, 1).float() / 255.0

    def encode(self, images: list) -> torch.Tensor:
        x = torch.stack([self._standardize(img) for img in images])  # (B, 3, S, S)
        return extract_flattened_patches(x, self.patch_size)  # (B, grid², 3·P·P); pool pinned to 1


def load_encoder(name: str, pool: int = 1, dtype: torch.dtype | None = None,
                 device: str | None = None, grid: int = 2,
                 patch_size: int = 32, img_size: int = 512):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    if name in ("patchify", "encoder_free"):  # no tower to load/place — just the resize+reshape
        return PatchifyEncoder(pool=pool, patch_size=patch_size, img_size=img_size)
    if name == "siglip_naflex":
        tower = Siglip2VisionModel.from_pretrained(_SIGLIP_NAFLEX_REPO, dtype=dtype).eval().to(device)
        for p in tower.parameters():
            p.requires_grad_(False)
        return Siglip2NaflexEncoder(tower=tower, processor=AutoImageProcessor.from_pretrained(_SIGLIP_NAFLEX_REPO),
                                    d_enc=tower.config.hidden_size, pool=pool)
    if name == "siglip":
        tower = SiglipVisionModel.from_pretrained(_SIGLIP_REPO, dtype=dtype).eval().to(device)
        for p in tower.parameters():
            p.requires_grad_(False)
        processor = AutoImageProcessor.from_pretrained(_SIGLIP_REPO)
        return SiglipAnyResEncoder(tower=tower, processor=processor, d_enc=tower.config.hidden_size,
                                   pool=pool, grid=grid)
    if name not in _REPOS:
        raise ValueError(f"unknown encoder {name!r}; choose from "
                         f"{list(_REPOS) + ['siglip', 'siglip_naflex', 'patchify']}")
    repo = _REPOS[name]
    full = AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype)
    tower = _resolve(full, _TOWER_PATH).eval().to(device)
    for p in tower.parameters():
        p.requires_grad_(False)
    # Image processor only: the full AutoProcessor also instantiates a video sub-processor
    # we never use; encode() only needs `pixel_values` + `image_grid_thw`.
    processor = AutoImageProcessor.from_pretrained(repo)
    d_enc = tower.config.out_hidden_size
    return Encoder(tower=tower, processor=processor, d_enc=d_enc, pool=pool)
