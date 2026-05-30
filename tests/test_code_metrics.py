from laguna_rlvr.visual.code_metrics import (
    code_validity_rate,
    compiles_python,
    is_valid,
    parses_html,
)


def test_parses_html_rejects_plain_text():
    assert parses_html("<div>hi</div>")
    assert not parses_html("just plain text, no tags")


def test_compiles_python():
    assert compiles_python("import matplotlib.pyplot as plt\nplt.plot([1, 2])")
    assert not compiles_python("def (:")  # syntax error


def test_is_valid_by_kind():
    assert is_valid("<p>x</p>", "html")
    assert not is_valid("def (:", "python")
    assert is_valid("anything", None)  # no code target -> not scored, treated valid


def test_validity_rate_scores_only_code_targets():
    rate = code_validity_rate(["<a>1</a>", "def (:", "plain"], ["html", "python", None])
    assert rate == 0.5  # html valid + python invalid; None skipped


def test_validity_rate_none_when_no_code_target():
    assert code_validity_rate(["a", "b"], [None, None]) is None
