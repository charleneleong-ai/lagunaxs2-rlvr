import pytest

from laguna_rlvr.synth.task import validate_task

ga = pytest.importorskip("general_agent")  # the installed env package


def test_builtin_tasks_are_self_consistent():
    # every builtin tier's gold solution must satisfy its own verifier
    assert all(validate_task(t) for t in ga._BUILTIN_TASKS)


def test_tool_signatures_extracted():
    sigs = ga._tool_signatures(ga._TOOLS)
    assert any("book_appointment" in s for s in sigs)
    assert any("list_services" in s for s in sigs)


def test_builtin_env_builds():
    assert len(ga.load_environment(source="builtin").eval_dataset) == 2
