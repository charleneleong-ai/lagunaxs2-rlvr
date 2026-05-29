from laguna_rlvr.synth.generate import is_learnable, parse_task, synthesize
from laguna_rlvr.synth.task import validate_task

_VALID = """Sure, here is a task.
DOMAIN: day_spa
TIER: 0
INSTRUCTION: Book a 'deep tissue' appointment.
[SCHEMA]
```python
from pydantic import BaseModel
class Appointment(BaseModel):
    service: str
```
[TOOLS]
```python
db = {"appointments": []}
def book_appointment(service):
    db["appointments"].append(Appointment(service=service))
```
[GOLD]
```python
book_appointment("deep tissue")
```
[VERIFIER]
```python
def verify():
    return any(a.service == "deep tissue" for a in db["appointments"])
```
"""

_INVALID_GOLD = _VALID.replace('book_appointment("deep tissue")', 'book_appointment("facial")')
_MISSING_SECTION = _VALID.split("[VERIFIER]")[0]


class TestParseTask:
    def test_extracts_all_sections(self):
        t = parse_task(_VALID)
        assert t is not None and t.domain == "day_spa" and t.tier == 0
        assert "deep tissue" in t.instruction and "book_appointment" in t.tools_code

    def test_missing_section_returns_none(self):
        assert parse_task(_MISSING_SECTION) is None


class TestSynthesize:
    def test_returns_a_self_consistent_task(self):
        task = synthesize(lambda _prompt: _VALID)
        assert task is not None and validate_task(task)

    def test_rejects_task_whose_gold_fails_its_verifier(self):
        # gold books the wrong service → verify() never True → no valid task after retries
        assert synthesize(lambda _prompt: _INVALID_GOLD, max_retries=2) is None


class TestIsLearnable:
    def _solver(self, pattern):
        it = iter(pattern * 10)
        return lambda _task: next(it)

    def test_partial_pass_is_learnable(self):
        assert is_learnable("t", self._solver([True, False, True, False]), samples=4) is True

    def test_always_solved_is_not_learnable(self):
        assert is_learnable("t", self._solver([True]), samples=4) is False

    def test_never_solved_is_not_learnable(self):
        assert is_learnable("t", self._solver([False]), samples=4) is False
