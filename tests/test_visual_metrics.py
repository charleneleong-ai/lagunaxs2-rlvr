from laguna_rlvr.visual.metrics import cer, score_predictions


def test_cer_zero_on_exact_match():
    assert cer("hello", "hello") == 0.0


def test_cer_one_on_full_mismatch():
    assert cer("", "abcd") == 1.0           # 4 insertions / 4 ref chars


def test_cer_partial():
    assert abs(cer("helo", "hello") - 0.2) < 1e-6   # 1 deletion / 5 ref chars


# --- score_predictions: adapter-free scoring core (shared by generation_metrics + Stage-0 baselines) ---

def test_score_predictions_scopes_wer_cer_to_ocr():
    # WER/CER only over OCR targets (kind None); code targets ride on code_valid/codebleu.
    out = score_predictions(["hello world", "<p>x</p>"], ["hello world", "<p>y</p>"], [None, "html"])
    assert out["val/metrics/wer"] == 0.0 and out["val/metrics/cer"] == 0.0  # OCR item is exact
    assert "val/metrics/code_valid" in out      # html item still scored for validity
    assert "val/metrics/codebleu" not in out    # no python target


def test_score_predictions_codebleu_and_validity_over_python():
    code = "def add(a, b):\n    c = a + b\n    return c"
    out = score_predictions([code], [code], ["python"])
    assert out["val/metrics/codebleu"] > 0.8 and out["val/metrics/code_valid"] == 1.0
    assert "val/metrics/wer" not in out         # no OCR target -> no transcription metric


def test_score_predictions_prefix_namespaces_keys():
    assert "eval/metrics/wer" in score_predictions(["x"], ["x"], [None], prefix="eval")
