from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoImageProcessor, AutoModelForImageTextToText

from laguna_rlvr.visual.projector import mean_pool

_REPOS = {"glm_ocr": "zai-org/GLM-OCR", "qwen3_vl": "Qwen/Qwen3-VL-4B-Instruct",
          "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct"}  # stronger general vision tower for the projector

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


def load_encoder(name: str, pool: int = 1, dtype: torch.dtype | None = None,
                 device: str | None = None) -> Encoder:
    if name not in _REPOS:
        raise ValueError(f"unknown encoder {name!r}; choose from {list(_REPOS)}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
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
