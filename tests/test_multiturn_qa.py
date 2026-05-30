from laguna_rlvr.visual.multiturn_qa import (
    Episode,
    QARef,
    load_manifest,
    read_question,
    run_qa,
    save_manifest,
    synthetic_episodes,
)


def _ep(needle_a: str, needle_b: str, kind_a: str = "python", kind_b: str = "html") -> Episode:
    return Episode(QARef("chartmimic", 0, needle_a, kind_a), QARef("design2code", 1, needle_b, kind_b))


def test_manifest_roundtrip(tmp_path):
    eps = [_ep("Sales 2024", "Welcome"), _ep("Q3 Revenue", "Pricing")]
    path = tmp_path / "qa.jsonl"
    save_manifest(eps, path)
    assert load_manifest(path) == eps  # dataclass equality, incl. kind=None handling


def test_read_question_is_kind_specific():
    assert "chart" in read_question("python").lower()
    assert "page" in read_question("html").lower() or "heading" in read_question("html").lower()
    assert read_question(None)  # non-empty fallback for needle-less kinds


def test_run_qa_scores_reading_and_recall():
    ep = _ep("Sales 2024", "Welcome Page")

    def run(_ep):  # r1 hits A, r2 misses B, r3 (recall) hits A — verbose replies, substring match
        return ["the title is Sales 2024", "some chart", "earlier it said Sales 2024"]

    m = run_qa(run, [ep])
    assert m["qa/metrics/accuracy"] == 0.5  # 1 of 2 reads
    assert m["qa/metrics/recall"] == 1.0


def test_synthetic_episodes_have_distinct_needles():
    eps = synthetic_episodes(n=3, seed=0)
    needles = [e.a.needle for e in eps] + [e.b.needle for e in eps]
    assert len(eps) == 3 and len(set(needles)) == len(needles)
