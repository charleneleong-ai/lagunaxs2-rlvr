# Audio Adapter — "Laguna gains ears" Design

**Date:** 2026-05-30
**Branch:** `feat/audio-adapter` (off `feat/visual-adapter-ocr-encoder`, reusing the visual-adapter stack)
**Context:** The visual adapter gave Laguna document understanding via a frozen-encoder → projector → frozen-LLM bridge. That pattern is **modality-agnostic** — the LLM consumes `inputs_embeds` and doesn't care what produced them. This spec extends it to **audio** (speech → text) by swapping the front-end encoder, reusing everything downstream.

> **Status (built 2026-05-30):** the modality-agnostic core was extracted to [`laguna_rlvr/mm/`](../../src/laguna_rlvr/mm/) (`model.ModalityAdapter`, `projector`, `metrics`, `seed`), so paths below that read `visual/…` for the *shared* pieces now live under `mm/`. The audio modality lands in its own package: [`audio/encoders.py`](../../src/laguna_rlvr/audio/encoders.py) (`load_audio_encoder`) + [`audio/data.py`](../../src/laguna_rlvr/audio/data.py) (`LibriSpeechASR`), wired into the trainer via `--modality audio`. `wer()` is in [`mm/metrics.py`](../../src/laguna_rlvr/mm/metrics.py). Overfit-one-batch proof: [`tests/test_audio_model.py`](../../tests/test_audio_model.py).

## Summary

A LLaVA-style audio adapter: `audio → [frozen Whisper encoder] → [trainable projector] → audio tokens → [frozen LLM] → text`. Only the projector trains. It is **purely additive** to the visual work — the projector, the `inputs_embeds` injection, the masked-CE training loop, the eval harness, and the GPU/quant-load path are all reused unchanged. The single new piece is an audio encoder wrapper; data is real speech with gold transcripts (self-verifying).

Debugged on the small Qwen base here; swappable to NVFP4 Laguna on the 80GB GPU, identical to the visual path.

## Why this is cheap (what's reused vs new)

| Layer | Source | Change |
|---|---|---|
| `Projector` (d_in→d_llm, + pooling) | `src/laguna_rlvr/visual/projector.py` | **reuse as-is** (audio `d_enc` is just a different `d_in`) |
| `VisualAdapter` (encode→project→prepend→frozen LLM→masked CE; `transcribe()`) | `src/laguna_rlvr/visual/model.py` | **reuse** — it depends only on an `Encoder` with `.encode()/.d_enc`, not on images |
| Train loop / CER-WER eval / GPU + quant-load path | `mm/train.py`, `mm/eval.py` | **reuse**, with `--modality audio` selecting the audio encoder + data |
| Config guardrails (`AdapterPlan`, `modality` field) | `src/laguna_rlvr/mm_adapter.py` | **reuse** — `modality.kind="audio"`, `encoder_id="openai/whisper-large-v3"` |
| **Audio encoder** | new: `src/laguna_rlvr/visual/audio_encoders.py` | the only genuinely new component (~40 lines) |
| **Audio data** | new: `visual/audio_data.py` | small real ASR slice (gold transcripts) |

The `Encoder` interface (`.encode(batch) -> (B, N/pool, d_enc)`, `.d_enc`, `.pool`, frozen `.tower`) is the contract; the audio encoder implements it so `VisualAdapter` and the trainer are untouched.

## Architecture & data flow

```
   speech.wav ──► [WhisperProcessor log-mel] ──► [frozen Whisper encoder] ──► (B, frames, d_enc)
                                                          │  optional mean-pool (frames are many: ~1500/30s)
                                                          ▼
                              [trainable Linear projector]  d_enc → d_llm
                                                          ▼
              prepend to "Transcribe the speech:" prompt embeds ──► [frozen LLM] ──► transcript
                                                                     Qwen3-0.6B (debug) → Laguna (80GB)
```

## Components

- **`audio_encoders.py`** — `load_audio_encoder(name="whisper_large", pool=k) -> Encoder`. Loads `openai/whisper-large-v3`, takes `model.get_encoder()` (frozen, `eval()`, `requires_grad_(False)`), exposes `WhisperProcessor` and `d_enc = encoder.config.d_model` (whisper-large-v3 = 1280; small = 768; tiny = 384 for CI). `.encode(audios)` runs the feature extractor (16 kHz log-mel) → encoder → `last_hidden_state` (B, ~1500, d_enc) → `mean_pool(·, k)` (reuse `projector.mean_pool`; pool hard, e.g. k=8–16, since frame count is large).
- **`audio_data.py`** — wrap a tiny real ASR set with gold transcripts, e.g. `hf-internal-testing/librispeech_asr_dummy` (~9 MB, fits the disk) → `(waveform, transcript)` pairs. Self-verifying (the dataset's gold transcript is the label). Synthetic TTS is a documented alternative for infinite/controlled data but adds a TTS dep — deferred.
- **Reused:** `VisualAdapter`, `train.py`, `eval.py`, `Projector`, `mm_adapter` guardrails.

## Evaluation

Primary metric **WER** (word error rate) — the speech analog of the visual CER. `jiwer` (already a dep) computes both; add a `wer()` next to `cer()` in `metrics.py`. Held-out ASR slice + (optionally) the Whisper-direct transcription as the zero-training baseline floor, mirroring the GLM-OCR baseline in the visual eval.

## Build sequence (mirrors the visual adapter)

1. `audio_encoders.py` + smoke (`load_audio_encoder('whisper_large').encode([wav])` → shape, record `d_enc`; CI uses `whisper_tiny`).
2. `audio_data.py` + a tiny ASR slice; `wer()` in `metrics.py`.
3. Generalize the trainer's encoder/data selection behind `--modality audio` (no change to `VisualAdapter`).
4. Overfit-one-batch test on Qwen3-0.6B (loss halves) — the wiring proof, identical shape to the visual test.
5. Short train + WER eval on the small base; then swap `--base` to Laguna on the 80GB GPU.

## Risks

- **Frame count**: Whisper emits ~1500 frames / 30 s → many tokens into the LLM. Pool aggressively (`k=8–16`) and/or cap audio length; same token-budget concern as vision, same `mean_pool` knob.
- **Encoder I/O differs**: waveforms + a feature extractor (log-mel, 16 kHz) instead of pixels — isolated entirely in `audio_encoders.py`.
- **Dep**: needs `librosa`/`soundfile` for audio loading (small). Whisper itself is already covered by `transformers`.
- Same VRAM rule: encoders are small and fit the 40 GB card; only the Laguna **decoder** needs 80 GB.

## Out of scope

Interleaved audio+vision in one prompt (additive later: tag token types); a shared cross-modal projector; the real Laguna run (80 GB).

**Speech output / TTS — deliberately deferred (decided 2026-05-30): the adapter stays STT-only.** The
thesis is *verifiable perception* (see/hear the artifact → act → verify); a coding agent rarely needs
to speak, so all three TTS flavors are out of scope: (1) TTS-synthesized ASR *training data* — the
cheapest, on-architecture extension if we ever want unlimited/domain-targeted speech (one light dep);
(2) an output *pipeline* (LLM text → off-the-shelf TTS) — pure plumbing, no model change; (3) native
speech *generation* (LLM emits audio-codec tokens → neural vocoder, Moshi/Qwen-Omni style) — a separate
research program that breaks the frozen-backbone design (trains the backbone + adds a codec/vocoder).
Revisit (1) first if the scope ever broadens past perception.

## References

- Reused stack: `src/laguna_rlvr/visual/{projector,model,train,eval}.py`, `src/laguna_rlvr/mm_adapter.py`.
- Pattern: LLaVA ([arXiv:2304.08485](https://arxiv.org/abs/2304.08485)); audio-LLM precedent: Qwen-Audio / SALMONN (frozen Whisper encoder + projector into an LLM).
