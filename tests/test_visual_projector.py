import torch
from laguna_rlvr.visual.projector import Projector, mean_pool


def test_linear_projector_maps_dims():
    p = Projector(d_in=16, d_out=32, kind="linear")
    out = p(torch.randn(2, 5, 16))          # (B, N, d_in)
    assert out.shape == (2, 5, 32)          # (B, N, d_out)


def test_mlp_projector_maps_dims():
    p = Projector(d_in=16, d_out=32, kind="mlp")
    assert p(torch.randn(2, 5, 16)).shape == (2, 5, 32)


def test_mean_pool_reduces_token_count():
    x = torch.randn(2, 8, 16)
    assert mean_pool(x, k=4).shape == (2, 2, 16)   # N 8 -> 2


def test_mean_pool_k1_is_identity():
    x = torch.randn(2, 7, 16)
    assert torch.equal(mean_pool(x, k=1), x)


def test_only_projector_params_train():
    p = Projector(d_in=16, d_out=32, kind="linear")
    assert all(param.requires_grad for param in p.parameters())
