from __future__ import annotations

import io
import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

# Two ASR sources, both gold-transcript (self-verifying — the transcript IS the label):
#  - dummy: a 73-clip smoke set (~9 MB, cached) with a positional 80/20 holdout. Fast for tests/CI.
#  - full:  real LibriSpeech with genuinely disjoint partitions (no leakage), streamed so only the
#           first `n` clips download. The real train/eval the dummy can only stand in for.
_DUMMY = "hf-internal-testing/librispeech_asr_dummy"
_FULL = "openslr/librispeech_asr"
_FULL_SPLITS = {"train": "train.100", "eval": "validation"}  # disjoint real LibriSpeech partitions
_SAMPLE_RATE = 16_000  # Whisper's input rate


def _decode(audio: dict) -> np.ndarray:
    """Decode one (possibly multi-channel) flac entry to a mono float32 16 kHz waveform.

    `decode=False` hands us the raw flac so soundfile decodes it — avoids datasets' newer hard
    dependency on torchcodec/ffmpeg just to turn a flac file into a waveform."""
    raw = audio["bytes"] or Path(audio["path"]).read_bytes()
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != _SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=_SAMPLE_RATE)
    return wav


@lru_cache(maxsize=1)
def _dummy_split():
    from datasets import Audio, load_dataset  # heavy; load only when the audio corpus is requested

    ds = load_dataset(_DUMMY, "clean", split="validation")
    return ds.cast_column("audio", Audio(decode=False))


@lru_cache(maxsize=4)
def _full_rows(split: str, n: int) -> tuple:
    """First `n` (waveform, text) pairs of a real LibriSpeech partition, streamed + decoded once.
    Cached per (split, n) so repeated builds in one process don't re-stream."""
    from datasets import Audio, load_dataset

    ds = load_dataset(_FULL, "clean", split=_FULL_SPLITS[split], streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    return tuple((_decode(ex["audio"]), ex["text"]) for ex in itertools.islice(ds, n))


class LibriSpeechASR(Dataset):
    """(waveform, gold-transcript) pairs from LibriSpeech — self-verifying.

    `source="dummy"` (default): 73-clip smoke set, positional `split` holdout ('train'=first 80%,
    'eval'=last 20%, 'all'). `source="full"`: real LibriSpeech, `split` selects a disjoint partition
    ('train'->train.100, 'eval'->validation); streamed, so pass `n` to cap how many clips to pull.
    """

    def __init__(self, n: int | None = None, split: str = "all",
                 source: str = "dummy", holdout: float = 0.2):
        if source == "full":
            if n is None:
                raise ValueError("source='full' is streamed; pass n to cap how many clips to pull")
            self.rows = _full_rows(split, n)
            self._decoded = True
        elif source == "dummy":
            ds = _dummy_split()
            cut = round(len(ds) * (1 - holdout))
            if split == "train":
                ds = ds.select(range(cut))
            elif split == "eval":
                ds = ds.select(range(cut, len(ds)))
            elif split != "all":
                raise ValueError(f"unknown split {split!r} (use 'train'/'eval'/'all')")
            self.rows = ds if n is None else ds.select(range(min(n, len(ds))))
            self._decoded = False
        else:
            raise ValueError(f"unknown source {source!r} (use 'dummy'/'full')")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[np.ndarray, str]:
        if self._decoded:
            return self.rows[i]
        row = self.rows[i]
        return _decode(row["audio"]), row["text"]
