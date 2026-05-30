from laguna_rlvr.visual.code_metrics import (
    code_validity_rate,
    codebleu_score,
    compiles_python,
    parses_html,
)


def test_parses_html_rejects_plain_text():
    assert parses_html("<div>hi</div>")
    assert not parses_html("just plain text, no tags")


def test_compiles_python():
    assert compiles_python("import matplotlib.pyplot as plt\nplt.plot([1, 2])")
    assert not compiles_python("def (:")  # syntax error
    assert not compiles_python("")  # empty -> no statements
    assert not compiles_python("# only a comment")  # parses but no statements, not real code


def test_validity_rate_scores_only_code_targets():
    rate = code_validity_rate(["<a>1</a>", "def (:", "plain"], ["html", "python", None])
    assert rate == 0.5  # html valid + python invalid; None skipped


def test_validity_rate_none_when_no_code_target():
    assert code_validity_rate(["a", "b"], [None, None]) is None


def test_codebleu_scores_python_only():
    # identical python (with dataflow) -> high; the html item is ignored (codebleu has no HTML grammar)
    code = "def add(a, b):\n    c = a + b\n    return c"
    score = codebleu_score([code, "<a>1</a>"], [code, "<a>1</a>"], ["python", "html"])
    assert score > 0.8


def test_codebleu_none_without_python():
    assert codebleu_score(["<a>1</a>"], ["<a>1</a>"], ["html"]) is None


def test_soft_scalar_anchor_matches_embedding_median():
    import torch

    from laguna_rlvr.visual.model import VisualAdapter

    adapter = object.__new__(VisualAdapter)  # bypass the heavy __init__ (no 33B load)
    adapter._emb_norm_median = torch.tensor(3.0)
    vis = torch.tensor([[[5.0, 0, 0, 0], [0, 1.0, 0, 0]]])  # token norms 5 and 1 (mean 3)
    out = adapter._anchor(vis)
    # mean token norm anchored to the median (3.0)...
    assert abs(out.flatten(0, 1).norm(dim=-1).mean().item() - 3.0) < 1e-4
    # ...and inter-token ratio preserved (single scalar): 5:1 stays 5:1
    norms = out.flatten(0, 1).norm(dim=-1)
    assert abs((norms[0] / norms[1]).item() - 5.0) < 1e-4


def test_embedding_norm_ratio():
    import torch

    from laguna_rlvr.visual.model import VisualAdapter

    adapter = object.__new__(VisualAdapter)  # bypass the heavy __init__ (no 33B load)
    adapter._emb_norm_median = torch.tensor(1.0)
    # ratio measures the RAW (anchor=False) projection -> 2 tokens each L2 norm sqrt(4)=2.0
    adapter._project = lambda _images, anchor=True: torch.ones(1, 2, 4)

    assert abs(adapter.embedding_norm_ratio([object()]) - 2.0) < 1e-5  # 2.0 / 1.0
    assert adapter.embedding_norm_ratio([]) is None


def test_generation_metrics_scopes_wer_cer_to_ocr():
    from laguna_rlvr.visual.metrics import generation_metrics

    class _Adapter:  # transcribe returns the OCR ref verbatim, ignoring the image
        def transcribe(self, _imgs, max_new_tokens=48):
            return ["hello world"]

    items = [(None, "hello world", "synthetic"),  # kind None (OCR) -> exact match, wer/cer 0
             (None, "<p>x</p>", "websight")]       # kind html -> excluded from wer/cer
    out = generation_metrics(_Adapter(), items)
    # wer/cer scored over the OCR item only (==0); had the html item leaked in, wer would be >0
    assert out["val/metrics/wer"] == 0.0 and out["val/metrics/cer"] == 0.0
    assert "val/metrics/code_valid" in out  # html item still scored for validity
