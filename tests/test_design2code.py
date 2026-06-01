from laguna_rlvr.visual.design2code import (_FINAL_WEIGHTS, _color, _match_blocks, _parse_rgb,
                                             _position, _text_f1)


def _block(text, box=(0, 0, 10, 10), color="rgb(0, 0, 0)"):
    return {"text": text, "box": list(box), "color": color}


def test_match_blocks_pairs_by_text_similarity():
    gen = [_block("alpha"), _block("zzz")]
    ref = [_block("alpha"), _block("beta")]
    matches = _match_blocks(gen, ref)
    assert len(matches) == 1 and matches[0][0]["text"] == "alpha"  # only alpha-alpha clears the threshold


def test_position_and_color_normalized_by_total_blocks():
    # 1 perfectly-aligned match out of 4 reference blocks -> score ~1/4, NOT 1.0 (the fixed artifact)
    gen = [_block("alpha")]
    ref = [_block("alpha"), _block("beta"), _block("gamma"), _block("delta")]
    matches = _match_blocks(gen, ref)
    denom = max(len(gen), len(ref))
    assert len(matches) == 1 and denom == 4
    assert abs(_position(matches, ref_diag=100.0, n_total=denom) - 0.25) < 1e-6
    assert abs(_color(matches, n_total=denom) - 0.25) < 1e-6


def test_position_color_empty_pages_are_perfect():
    assert _position([], ref_diag=100.0, n_total=0) == 1.0
    assert _color([], n_total=0) == 1.0


def test_parse_rgb():
    assert _parse_rgb("rgb(34, 34, 34)") == (34, 34, 34)
    assert _parse_rgb("rgba(10,20,30,.5)") == (10, 20, 30)
    assert _parse_rgb("transparent") == (0, 0, 0)


def test_text_f1_endpoints():
    assert _text_f1([_block("hello world")], [_block("hello world")]) == 1.0
    assert _text_f1([_block("alpha")], [_block("beta")]) == 0.0


def test_final_weights_downweight_visual_sim():
    assert abs(sum(_FINAL_WEIGHTS.values()) - 1.0) < 1e-9
    assert _FINAL_WEIGHTS["visual_sim"] == min(_FINAL_WEIGHTS.values())  # saturated -> least weight
