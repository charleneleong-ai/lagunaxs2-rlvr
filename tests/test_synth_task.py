from laguna_rlvr.synth.task import Task, run_solution, validate_task

_DAY_SPA = Task(
    domain="day_spa",
    tier=0,
    schema_code="from pydantic import BaseModel\nclass Appointment(BaseModel):\n    service: str",
    tools_code=(
        'db = {"appointments": [], "services": ["swedish", "deep tissue", "facial"]}\n'
        "def list_services():\n    return list(db['services'])\n"
        "def book_appointment(service):\n    db['appointments'].append(Appointment(service=service))"
    ),
    instruction="Book a 'deep tissue' appointment.",
    gold_solution='book_appointment("deep tissue")',
    verifier_code="def verify():\n    return any(a.service == 'deep tissue' for a in db['appointments'])",
)


def test_gold_solution_satisfies_verifier():
    assert validate_task(_DAY_SPA) is True


def test_wrong_solution_fails_verifier():
    assert run_solution(_DAY_SPA, 'book_appointment("facial")') is False


def test_broken_code_is_invalid_not_crash():
    bad = Task(**{**_DAY_SPA.to_dict(), "gold_solution": "book_appointment( syntax error"})
    assert validate_task(bad) is False
