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


def test_run_benchmarks_passes_run_only_to_scorers_that_accept_it(monkeypatch):
    seen = {}

    def with_run(adapter, items, run=None, step=None):
        seen["with"] = run
        return {"a/metrics/x": 1.0}

    def without_run(adapter, items):
        seen["without"] = "called"
        return {"b/metrics/x": 1.0}

    monkeypatch.setattr(benchmarks, "BENCHMARKS", {
        "a": lambda n: ([0] * n, with_run),
        "b": lambda n: ([0] * n, without_run),
    })
    class _Run:
        def log(self, d, step=None):
            pass

    sentinel = _Run()
    benchmarks.run_benchmarks(adapter=object(), names=["a", "b"], n=2, run=sentinel, step=7)
    assert seen["with"] is sentinel and seen["without"] == "called"  # run injected only where accepted
