from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoImageProcessor, AutoModelForImageTextToText, SiglipVisionModel

from laguna_rlvr.visual.projector import mean_pool

_REPOS = {"glm_ocr": "zai-org/GLM-OCR", "qwen3_vl": "Qwen/Qwen3-VL-4B-Instruct",
          "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct"}  # stronger general vision tower for the projector
# SigLIP 2: same so400m backbone as laguna-vision's encoder, retrained with caption + self-distill +
# dense losses -> better localized features (OCR/UI). Drop-in (same SiglipVisionModel path). The
# naflex variant (native aspect-ratio / variable resolution) could later replace the AnyRes tiling.
_SIGLIP_REPO = "google/siglip2-so400m-patch16-384"


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


def load_encoder(name: str, pool: int = 1, dtype: torch.dtype | None = None,
                 device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    if name == "siglip":
        tower = SiglipVisionModel.from_pretrained(_SIGLIP_REPO, dtype=dtype).eval().to(device)
        for p in tower.parameters():
            p.requires_grad_(False)
        processor = AutoImageProcessor.from_pretrained(_SIGLIP_REPO)
        return SiglipAnyResEncoder(tower=tower, processor=processor, d_enc=tower.config.hidden_size, pool=pool)
    if name not in _REPOS:
        raise ValueError(f"unknown encoder {name!r}; choose from {list(_REPOS) + ['siglip']}")
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
