import pytest

from laguna_rlvr.visual import benchmarks


def test_registry_covers_the_wired_suite():
    assert set(benchmarks.BENCHMARKS) == {
        "ocrbench", "mmmu", "mathvista", "design2code", "screenspot_v2", "mmdu"}
    assert benchmarks.DEFAULT_SUITE == list(benchmarks.BENCHMARKS)
    assert all(callable(v) for v in benchmarks.BENCHMARKS.values())


def test_run_benchmarks_aggregates_and_logs(monkeypatch):
    class _Run:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step=None):
            self.logged.append((metrics, step))

    # a fake registry entry: dataset is a plain list, scorer returns canned metrics
    monkeypatch.setattr(benchmarks, "BENCHMARKS", {
        "fake": lambda n: (list(range(n)), lambda adapter, items: {
            "fake/metrics/accuracy": len(items) / 10}),
    })
    run = _Run()
    out = benchmarks.run_benchmarks(adapter=object(), names=["fake"], n=5, run=run, step=42)
    assert out == {"fake/metrics/accuracy": 0.5}
    assert run.logged == [({"fake/metrics/accuracy": 0.5}, 42)]


def test_run_benchmarks_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown benchmark"):
        benchmarks.run_benchmarks(adapter=object(), names=["nope"], n=1)


def test_run_benchmarks_skips_a_failing_benchmark(monkeypatch):
    def _boom(n):
        raise RuntimeError("dataset download failed")

    monkeypatch.setattr(benchmarks, "BENCHMARKS", {
        "bad": _boom,
        "good": lambda n: ([0] * n, lambda adapter, items: {"good/metrics/accuracy": 1.0}),
    })
    # the failing benchmark is skipped; the good one still scores
    out = benchmarks.run_benchmarks(adapter=object(), names=["bad", "good"], n=3)
    assert out == {"good/metrics/accuracy": 1.0}
