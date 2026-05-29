import torch

from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import VisualAdapter

BASE = "Qwen/Qwen3-0.6B"


def test_overfit_one_batch_drives_loss_down():
    """Train ONLY the projector on a single batch; loss must fall sharply -> wiring works."""
    enc = load_encoder("glm_ocr", pool=4)
    adapter = VisualAdapter(encoder=enc, base_llm=BASE, projector_kind="linear")
    ds = SyntheticOCR(texts=["hello 42", "total 7"], seed=0)
    images = [ds[0][0], ds[1][0]]
    labels = [ds[0][1], ds[1][1]]

    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-3)
    first = last = None
    for _ in range(30):
        loss = adapter(images, labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5  # projector alone learns the mapping on one batch
