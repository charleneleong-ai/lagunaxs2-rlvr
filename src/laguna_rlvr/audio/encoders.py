from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from laguna_rlvr.mm.projector import mean_pool

WHISPER_REPOS = {
    "whisper_tiny": "openai/whisper-tiny",          # 384-d, CI/smoke only
    "whisper_base": "openai/whisper-base",          # 512-d
    "whisper_small": "openai/whisper-small",        # 768-d
    "whisper_large": "openai/whisper-large-v3",     # 1280-d, the default speech front-end
}
_SAMPLE_RATE = 16_000  # Whisper's fixed input rate; the feature extractor resamples to it


@dataclass
class AudioEncoder:
    """Frozen Whisper encoder behind the same `.encode`/`.d_enc` contract as the visual `Encoder`,
    so the shared `ModalityAdapter` and trainer consume it unchanged; only the I/O here is audio."""

    tower: torch.nn.Module
    processor: object  # WhisperProcessor (16 kHz log-mel feature extractor)
    d_enc: int
    pool: int

    @torch.no_grad()
    def encode(self, audios: list) -> torch.Tensor:
        """Frozen Whisper encoder features for `audios` -> (B, frames/pool, d_enc).

        Each `audios[i]` is a 1-D 16 kHz waveform. The feature extractor pads every clip to
        Whisper's fixed 30 s window, so the encoder emits a constant ~1500 frames, mean-pooled
        by `pool` (large frame count, so audio pools harder than vision).
        """
        p = next(self.tower.parameters())
        feats = self.processor.feature_extractor(
            audios, sampling_rate=_SAMPLE_RATE, return_tensors="pt")
        out = self.tower(feats["input_features"].to(device=p.device, dtype=p.dtype))
        return mean_pool(out.last_hidden_state, self.pool)  # (B, ~1500/pool, d_enc)


def load_audio_encoder(name: str = "whisper_large", pool: int = 8,
                       dtype: torch.dtype | None = None, device: str | None = None) -> AudioEncoder:
    if name not in WHISPER_REPOS:
        raise ValueError(f"unknown audio encoder {name!r}; choose from {list(WHISPER_REPOS)}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    repo = WHISPER_REPOS[name]
    tower = WhisperForConditionalGeneration.from_pretrained(repo, dtype=dtype).get_encoder()
    tower = tower.eval().to(device)
    for p in tower.parameters():
        p.requires_grad_(False)
    processor = WhisperProcessor.from_pretrained(repo)
    return AudioEncoder(tower=tower, processor=processor, d_enc=tower.config.d_model, pool=pool)
