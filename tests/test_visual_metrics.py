from laguna_rlvr.visual.metrics import cer


def test_cer_zero_on_exact_match():
    assert cer("hello", "hello") == 0.0


def test_cer_one_on_full_mismatch():
    assert cer("", "abcd") == 1.0           # 4 insertions / 4 ref chars


def test_cer_partial():
    assert abs(cer("helo", "hello") - 0.2) < 1e-6   # 1 deletion / 5 ref chars
