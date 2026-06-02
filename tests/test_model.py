import torch
from types import SimpleNamespace

from laguna_rlvr.visual.model import VisualAdapter


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
