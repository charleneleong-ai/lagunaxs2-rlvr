import pytest
import torch
from transformers import AutoTokenizer

from laguna_rlvr.seed import seed_everything
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import IMAGE_TOKEN, VisualAdapter

BASE = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def adapter():
    seed_everything(42)  # deterministic projector init; module-scoped so the model loads once
    return VisualAdapter(encoder=load_encoder("glm_ocr", pool=4), base_llm=BASE, projector_kind="linear")


def test_overfit_one_batch_drives_loss_down(adapter):
    """Train ONLY the projector on a single batch; loss must fall sharply -> wiring works."""
    ds = SyntheticOCR(texts=["hello 42", "total 7"], seed=0)
    images, labels = [ds[0][0], ds[1][0]], [ds[0][1], ds[1][1]]
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-3)
    first = last = None
    for _ in range(150):  # seed-42 init learns slowly; 150 steps clears the >50%-drop bar decisively
        loss = adapter(images, labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5  # projector alone learns the mapping on one batch


class TestImageToken:
    """The <image> marker: subtoken-averaged init (Laguna §4.1.1) + splice into inputs_embeds."""

    def test_subtoken_avg_init(self, adapter):
        # <image> embedding == mean of the embeddings of its pre-add subtokens. (If the base already
        # had <image> as one token, sub_ids is that single row and the equality holds trivially.)
        fresh = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        sub_ids = fresh(IMAGE_TOKEN, add_special_tokens=False).input_ids
        emb = adapter.llm.get_input_embeddings().weight.data
        assert torch.allclose(emb[adapter.image_token_id], emb[sub_ids].mean(dim=0), atol=1e-5)

    def test_splice_replaces_marker_with_vision_tokens(self, adapter):
        d = adapter.llm.config.hidden_size
        nv = 5
        vis = torch.zeros(1, nv, d, device=adapter.llm.device, dtype=adapter.llm.dtype)
        text = f"before {IMAGE_TOKEN} after"
        n_text = adapter.tok(text, return_tensors="pt").input_ids.shape[1]
        out = adapter._embed_with_vision(text, vis)
        assert out.shape == (1, n_text - 1 + nv, d)  # the one marker slot -> nv vision slots

    def test_splice_requires_exactly_one_marker(self, adapter):
        vis = torch.zeros(1, 3, adapter.llm.config.hidden_size,
                          device=adapter.llm.device, dtype=adapter.llm.dtype)
        with pytest.raises(ValueError):
            adapter._embed_with_vision("no marker here", vis)
