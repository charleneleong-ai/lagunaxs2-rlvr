import torch

from laguna_rlvr.audio.data import LibriSpeechASR
from laguna_rlvr.audio.encoders import load_audio_encoder
from laguna_rlvr.mm.model import ModalityAdapter
from laguna_rlvr.mm.seed import seed_everything

BASE = "Qwen/Qwen3-0.6B"
_PROMPT = "Transcribe the speech:"


def test_overfit_one_batch_drives_loss_down():
    """Train ONLY the projector on a single speech batch; loss must fall sharply -> wiring works.

    Same proof as the visual overfit test, swapping the front-end encoder + data — the point of the
    modality-agnostic core: nothing downstream of the encoder changes between vision and audio."""
    seed_everything(0)  # the projector init is random; seed so the overfit is deterministic
    enc = load_audio_encoder("whisper_tiny", pool=8)  # tiny encoder: fast to download in CI
    adapter = ModalityAdapter(enc, base_llm=BASE, projector_kind="linear", prompt=_PROMPT)
    ds = LibriSpeechASR(n=2)
    audios = [ds[0][0], ds[1][0]]
    labels = [ds[0][1], ds[1][1]]

    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-3)
    first = last = None
    for _ in range(60):
        loss = adapter(audios, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5  # projector alone learns the speech->text mapping on one batch
