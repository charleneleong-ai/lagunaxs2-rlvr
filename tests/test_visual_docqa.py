from laguna_rlvr.visual.docqa import _norm, make_items


def test_make_items_deterministic_and_gold_in_render():
    a = make_items(3, seed=0)
    b = make_items(3, seed=0)
    assert len(a) == 3
    assert [x["answer"] for x in a] == [x["answer"] for x in b]  # seeded determinism
    for item in a:  # the question's gold value is actually present in the rendered lines
        assert any(item["answer"] in line for line in item["lines"])


def test_norm_enables_lenient_value_match():
    assert _norm("$1,240.50") == "124050"
    assert _norm("$1,240.50") in _norm("The total is $1,240.50.")
