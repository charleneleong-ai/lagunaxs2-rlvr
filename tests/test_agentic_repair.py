import pytest

ar = pytest.importorskip("agentic_repair")  # the installed env package


class TestInjectBug:
    _TESTS = ["assert add(2, 3) == 5", "assert add(0, 0) == 0"]

    def test_injects_a_failing_bug(self):
        buggy = ar.inject_bug("def add(a, b): return a + b", self._TESTS, "", timeout=5.0)
        assert buggy is not None and buggy != "def add(a, b): return a + b"

    def test_returns_none_when_no_mutation_applies(self):
        # no mutatable operator → no bug can be injected
        assert ar.inject_bug("def const(): return 7", ["assert const() == 7"], "", timeout=5.0) is None


def test_builtin_repair_env_builds_offline():
    env = ar.load_environment(source="builtin")
    assert len(env.eval_dataset) == 1


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        ar._load_repair_tasks("nope", 1, 0, 5.0)
