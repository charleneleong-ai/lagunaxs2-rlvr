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


def test_resampler_emits_n_queries_tokens():
    p = Projector(d_in=16, d_out=32, kind="resampler", n_queries=64)
    out = p(torch.randn(2, 300, 16))             # 300 input patches -> fixed n_queries
    assert out.shape == (2, 64, 32)


def test_resampler_defaults_to_256_queries():
    p = Projector(d_in=16, d_out=32, kind="resampler")
    assert p(torch.randn(1, 300, 16)).shape == (1, 256, 32)


def test_load_compatible_transfers_machinery_and_resizes_query_bank():
    src = Projector(16, 32, kind="resampler", n_queries=4)
    dst = Projector(16, 32, kind="resampler", n_queries=8)
    skipped = dst.load_compatible(src.state_dict())   # warm-start across a query-bank resize
    assert skipped == ["net.query"]                   # only the resized bank is left at init
    assert torch.equal(dst.net.kv.weight, src.net.kv.weight)  # cross-attn machinery transferred
    assert dst.net.query.shape == (8, 32)             # new bank keeps its larger size


def test_load_compatible_same_shape_loads_everything():
    src = Projector(16, 32, kind="resampler", n_queries=8)
    dst = Projector(16, 32, kind="resampler", n_queries=8)
    assert dst.load_compatible(src.state_dict()) == []
    assert torch.equal(dst.net.query, src.net.query)
