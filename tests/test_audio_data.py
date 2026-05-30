from laguna_rlvr.audio.data import LibriSpeechASR


def test_dummy_train_eval_splits_partition_without_leakage():
    """The positional holdout must cover the whole dummy set with disjoint train/eval — otherwise
    'held-out' eval would leak training clips."""
    full, train, ev = (LibriSpeechASR(split=s) for s in ("all", "train", "eval"))
    assert len(train) + len(ev) == len(full)
    assert 0 < len(ev) < len(train)
    train_texts = {train[i][1] for i in range(len(train))}
    eval_texts = {ev[i][1] for i in range(len(ev))}
    assert train_texts.isdisjoint(eval_texts)
