import torch
import torch.nn as nn
from types import SimpleNamespace

from laguna_rlvr.visual.model import VisualAdapter, _lora_targets, _unfreeze_top_k, _trainable_fast_slow


def _stub_llm(W):
    """Position-wise-independent LM: logits[b,t] = emb[b,t] @ W. Padding/batching can't change a real
    token's logits, so a correct batched loss must equal the per-example loop to fp precision."""
    return lambda *, inputs_embeds, attention_mask: SimpleNamespace(logits=inputs_embeds @ W)


def _example(seq_len: int, prompt_len: int, D: int, V: int):
    s = torch.randn(seq_len, D)
    y = torch.full((seq_len,), -100, dtype=torch.long)
    y[prompt_len:] = torch.randint(0, V, (seq_len - prompt_len,))  # prompt masked, answer supervised
    return s, y


def test_batched_lm_loss_equals_per_example_loop():
    torch.manual_seed(0)
    D, V = 8, 50
    W = torch.randn(D, V)
    llm = _stub_llm(W)
    seqs, labels = zip(*[_example(L, p, D, V) for L, p in ((5, 2), (8, 3), (3, 1))])  # variable lengths

    batched = VisualAdapter._batched_lm_loss(list(seqs), list(labels), llm)

    refs = []  # the loop it replaces: per-example mean CE, then averaged
    for s, y in zip(seqs, labels):
        logits = llm(inputs_embeds=s[None], attention_mask=torch.ones(1, s.shape[0])).logits[:, :-1]
        refs.append(torch.nn.functional.cross_entropy(logits.reshape(-1, V), y[None][:, 1:].reshape(-1),
                                                      ignore_index=-100))
    assert torch.allclose(batched, torch.stack(refs).mean(), atol=1e-5), (batched, torch.stack(refs).mean())


def test_batched_lm_loss_padding_is_inert():
    """A short example padded next to a long one yields the same loss as run alone (padding ignored)."""
    torch.manual_seed(1)
    D, V = 6, 30
    W = torch.randn(D, V)
    llm = _stub_llm(W)
    short = _example(3, 1, D, V)
    alone = VisualAdapter._batched_lm_loss([short[0]], [short[1]], llm)
    long = _example(9, 4, D, V)
    # same short example, now batched with a longer one -> its per-example loss must be unchanged
    batched_pair = VisualAdapter._batched_lm_loss([short[0], long[0]], [short[1], long[1]], llm)
    long_alone = VisualAdapter._batched_lm_loss([long[0]], [long[1]], llm)
    assert torch.allclose(batched_pair, torch.stack([alone, long_alone]).mean(), atol=1e-5)


# --- decoder-unfreeze targeting (Laguna MoE: routed experts are batched nn.Parameter, not Linear) ---

class _Experts(nn.Module):  # LagunaExperts: routed weights are 3D Parameters, NOT modules -> not LoRA-able
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 2, 2))
        self.down_proj = nn.Parameter(torch.zeros(2, 2, 2))


class _MLP(nn.Module):  # LagunaMLP (shared expert / dense layer): real Linears
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(2, 2, bias=False)
        self.up_proj = nn.Linear(2, 2, bias=False)
        self.down_proj = nn.Linear(2, 2, bias=False)


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _Experts()
        self.gate = nn.Module()  # LagunaTopKRouter: router weight is a bare Parameter, not a Linear
        self.gate.weight = nn.Parameter(torch.zeros(2, 2))
        self.shared_experts = _MLP()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, n, nn.Linear(2, 2, bias=False))
        self.mlp = _MoE()
        self.input_layernorm = nn.LayerNorm(2)


class _LLM(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Layer() for _ in range(n_layers))


def test_lora_targets_attention_only_by_default():
    assert _lora_targets(include_ffn=False) == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_lora_targets_add_ffn_linears_for_moe():
    t = _lora_targets(include_ffn=True)
    assert set(t) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    # these names only resolve to LagunaMLP Linears (dense MLP + shared_experts); routed experts are
    # Parameters named gate_up_proj/down_proj inside `.experts.` and stay frozen (PEFT skips non-modules)


def test_top_k_unfreezes_attn_shared_router_norms_only_in_top_layers():
    llm = _LLM(4)
    for p in llm.parameters():
        p.requires_grad_(False)
    n = _unfreeze_top_k(llm, 2)

    def grad(layer, name):
        return dict(layer.named_parameters())[name].requires_grad

    top, bottom = llm.model.layers[2:], llm.model.layers[:2]
    for layer in top:  # attention, shared expert, router, norm: trainable
        assert grad(layer, "self_attn.q_proj.weight")
        assert grad(layer, "mlp.shared_experts.down_proj.weight")
        assert grad(layer, "mlp.gate.weight")
        assert grad(layer, "input_layernorm.weight")
        # routed experts (the memory bomb) stay frozen even in the unfrozen layers
        assert not grad(layer, "mlp.experts.gate_up_proj")
        assert not grad(layer, "mlp.experts.down_proj")
    for layer in bottom:  # untouched below the top-k window
        assert not grad(layer, "self_attn.q_proj.weight")
        assert not grad(layer, "mlp.shared_experts.down_proj.weight")
    assert n == sum(p.requires_grad for p in llm.parameters())


def test_trainable_fast_slow_splits_lora_from_unfrozen_base():
    named = [("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", SimpleNamespace(requires_grad=True)),
             ("model.layers.39.self_attn.q_proj.weight", SimpleNamespace(requires_grad=True)),   # unfrozen base (top-k)
             ("model.layers.0.self_attn.q_proj.weight", SimpleNamespace(requires_grad=False))]   # frozen
    fast, slow = _trainable_fast_slow(named)
    assert [n for n, _ in fast] == ["base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]
    assert [n for n, _ in slow] == ["model.layers.39.self_attn.q_proj.weight"]
