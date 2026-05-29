import pytest

cs = pytest.importorskip("code_smoke")  # the installed env package


def test_builtin_tasks_are_prompt_tests_setup_triples():
    tasks = cs._load_tasks("builtin", None, 0)
    assert len(tasks) == 3
    assert all(len(t) == 3 and isinstance(t[1], list) for t in tasks)  # (prompt, tests, setup)


def test_non_builtin_source_is_treated_as_hf_id():
    # builtin/mbpp are special-cased; anything else is an HF dataset id, so a bogus one errors out.
    with pytest.raises(Exception):
        cs._load_tasks("definitely/not-a-real-dataset-xyz", 1, 0)


def test_builtin_env_builds_offline():
    env = cs.load_environment(source="builtin")
    assert len(env.eval_dataset) == 3
