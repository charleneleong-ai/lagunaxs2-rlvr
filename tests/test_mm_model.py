import torch

from laguna_rlvr.mm.model import ModalityAdapter
from laguna_rlvr.mm.seed import seed_everything
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder

BASE = "Qwen/Qwen3-0.6B"


def test_overfit_one_batch_drives_loss_down():
    """Train ONLY the projector on a single batch; loss must fall sharply -> wiring works."""
    seed_everything(0)  # the projector init is random; seed so the overfit is deterministic
    enc = load_encoder("glm_ocr", pool=4)
    adapter = ModalityAdapter(encoder=enc, base_llm=BASE, projector_kind="linear")
    ds = SyntheticOCR(texts=["hello 42", "total 7"], seed=0)
    images = [ds[0][0], ds[1][0]]
    labels = [ds[0][1], ds[1][1]]

    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-3)
    first = last = None
    for _ in range(60):
        loss = adapter(images, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5  # projector alone learns the mapping on one batch
