import pytest

cs = pytest.importorskip("code_smoke")  # the installed env package


def test_builtin_tasks_are_prompt_tests_setup_triples():
    tasks = cs._load_tasks("builtin", None, 0)
    assert len(tasks) == 3
    assert all(len(t) == 3 and isinstance(t[1], list) for t in tasks)  # (prompt, tests, setup)


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        cs._load_tasks("nope", 1, 0)


def test_builtin_env_builds_offline():
    env = cs.load_environment(source="builtin")
    assert len(env.eval_dataset) == 3
