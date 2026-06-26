"""Encoder-free VLM path (Gemma 4 / Fuyu style): PatchifyEncoder + patch_embed projector.

Validates the embedder in isolation (patchify correctness, projection shape, factorized positions,
grad flow) and end-to-end through VisualAdapter — the raw-pixel path must learn to feed the frozen LLM.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from laguna_rlvr.seed import seed_everything
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.encoders import PatchifyEncoder, extract_flattened_patches, load_encoder
from laguna_rlvr.visual.model import VisualAdapter
from laguna_rlvr.visual.projector import PatchEmbedder, Projector
from laguna_rlvr.visual.train import _assert_patchify_fits

BASE = "Qwen/Qwen3-0.6B"


def _img(w: int, h: int) -> Image.Image:
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


class TestPatchify:
    """Raw-pixel patch extraction: the loop-free reshape→permute→reshape."""

    def test_row_major_patch_values(self):
        # 4×4 single-channel image, 2×2 patches -> 4 patches, each row-major [tl, tr, bl, br].
        x = torch.arange(16).float().reshape(1, 1, 4, 4)
        out = extract_flattened_patches(x, patch_size=2)
        expected = torch.tensor([[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]]).float()
        assert torch.equal(out, expected)


class TestPatchifyEncoder:
    """The encoder-free 'tower': resize+crop+patchify, Encoder-compatible interface."""

    def test_encode_shape_and_dims(self):
        enc = PatchifyEncoder()
        out = enc.encode([_img(640, 480), _img(200, 900)])  # mixed sizes standardize to one shape
        assert out.shape == (2, 256, 3072)  # 16×16 patches, 3·32·32-dim each
        assert enc.d_enc == 3072 and enc.grid == 16

    @pytest.mark.parametrize("img_size,patch_size,grid,d_enc", [
        (512, 32, 16, 3072),   # default: coarse
        (512, 16, 32, 768),    # finer grid, smaller per-patch dim (higher-res reading test)
        (768, 32, 24, 3072),   # bigger canvas, same patch
    ])
    def test_resolution_knobs_set_grid_and_dims(self, img_size, patch_size, grid, d_enc):
        enc = load_encoder("patchify", patch_size=patch_size, img_size=img_size)
        out = enc.encode([_img(640, 480)])
        assert enc.grid == grid and enc.d_enc == d_enc
        assert out.shape == (1, grid * grid, d_enc)

    @pytest.mark.parametrize("grid,fits", [(16, True), (24, True), (32, False), (28, False)])
    def test_seq_cap_guard_rejects_token_blowup(self, grid, fits):
        # A grid whose grid² vision tokens crowd out the label budget can never train (loss=0); the
        # guard must reject it loudly. patch16@512 → grid 32 = 1024 tokens is the case that bit us.
        if fits:
            _assert_patchify_fits(grid)
        else:
            with pytest.raises(SystemExit, match="loss is 0"):
                _assert_patchify_fits(grid)

    def test_small_image_upscaled_and_normalized(self):
        # A 10×10 image must upscale (shorter side -> 512), crop to exactly 512×512, pixels in [0, 1].
        t = PatchifyEncoder()._standardize(_img(10, 10))
        assert t.shape == (3, 512, 512) and 0.0 <= t.min() and t.max() <= 1.0

    def test_pool_above_one_is_rejected(self):
        with pytest.raises(ValueError):  # pooling raw patches scrambles the positional grid
            PatchifyEncoder(pool=4)

    def test_load_encoder_returns_patchifier_without_a_tower(self):
        enc = load_encoder("patchify")
        assert isinstance(enc, PatchifyEncoder) and not hasattr(enc, "tower")
        assert load_encoder("encoder_free").d_enc == 3072  # alias


class TestPatchEmbedder:
    """The trainable embedder body: LN→Linear→LN→+pos→LN→connector."""

    def test_maps_patch_dim_to_hidden(self):
        out = PatchEmbedder(patch_dim=3072, d_out=64)(torch.randn(2, 256, 3072))
        assert out.shape == (2, 256, 64)

    def test_positions_are_factorized_row_plus_col(self):
        # Grid side is read from N=9 -> 3×3; pos(i,j) must equal row[i]+col[j].
        emb = PatchEmbedder(patch_dim=4, d_out=8)
        pos = emb._positions(9, torch.float32)
        for i in range(3):
            for j in range(3):
                assert torch.allclose(pos[i * 3 + j], emb.row_emb[i] + emb.col_emb[j])

    def test_non_square_token_count_is_rejected(self):
        with pytest.raises(ValueError):  # √N must be integer — positions are a square grid
            PatchEmbedder(patch_dim=4, d_out=8)._positions(10, torch.float32)

    def test_position_breaks_patch_order_symmetry(self):
        # Two identical patches at different slots must differ after the embedder (position injected).
        out = PatchEmbedder(patch_dim=4, d_out=8)(torch.ones(1, 4, 4))
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_grad_flows_to_every_trainable_block(self):
        emb = PatchEmbedder(patch_dim=4, d_out=8)
        emb(torch.randn(1, 4, 4)).sum().backward()
        for name in ("fc.weight", "row_emb", "col_emb", "connector.weight"):
            g = dict(emb.named_parameters())[name].grad
            assert g is not None and g.abs().sum() > 0

    def test_projector_kind_wires_the_embedder(self):
        assert isinstance(Projector(d_in=3072, d_out=64, kind="patch_embed").net, PatchEmbedder)


@pytest.fixture(scope="module")
def ef_adapter():
    seed_everything(42)
    return VisualAdapter(encoder=load_encoder("patchify"), base_llm=BASE, projector_kind="patch_embed")


def test_encoder_free_overfits_one_batch(ef_adapter):
    """Train ONLY the embedder on one batch through the frozen LLM; loss must fall -> the raw-pixel
    path is wired end-to-end (patchify -> embedder -> splice -> frozen decoder)."""
    ds = SyntheticOCR(texts=["hello 42", "total 7"], seed=0)
    images, labels = [ds[0][0], ds[1][0]], [ds[0][1], ds[1][1]]
    opt = torch.optim.AdamW(ef_adapter.trainable_parameters(), lr=1e-3)
    first = last = None
    for _ in range(150):
        loss = ef_adapter(images, labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5
