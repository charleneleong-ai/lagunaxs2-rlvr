"""General-agent task-corpus format + self-consistency validator.

A task (one row of a general-agent-style corpus) bundles a domain's:
  - schema_code   : Pydantic entity models
  - tools_code    : a module-level `db` state + tool functions that mutate it
  - instruction   : natural-language goal
  - gold_solution : a sequence of tool calls that should satisfy the goal
  - verifier_code : `def verify() -> bool` reading `db` to check completion
  - tier          : difficulty 0-4

The synthesizer is only allowed to KEEP a task if it is *self-consistent*: assembling
schema + tools + gold + verifier and running it must make `verify()` return True. That
execution check (in a subprocess, like code_exec) is the corpus's correctness backbone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from laguna_rlvr.code_exec import run_python

_HARNESS = "import sys as _s; _s.exit(0 if verify() else 1)"


@dataclass(frozen=True, slots=True)
class Task:
    domain: str
    tier: int
    schema_code: str
    tools_code: str
    instruction: str
    gold_solution: str
    verifier_code: str

    def to_dict(self) -> dict:
        return asdict(self)


def assemble_program(task: Task, solution: str | None = None) -> str:
    """schema + tools + (a candidate solution, default the gold) + verifier + exit-on-verify."""
    return "\n\n".join([task.schema_code, task.tools_code,
                        task.gold_solution if solution is None else solution,
                        task.verifier_code, _HARNESS])


def run_capturing(task: Task, solution: str | None = None, timeout: float = 5.0) -> tuple[bool, str]:
    """Run the solution once; return (verifier_passed, captured stdout+stderr).

    One subprocess yields both the score (verify() harness exit code) and the agent's observation
    (whatever the tool calls printed) — so a multi-turn env needn't run the program twice per turn.
    """
    r = run_python(assemble_program(task, solution), timeout)
    if r is None:
        return False, "(timed out)"
    return r.returncode == 0, (r.stdout + r.stderr)


def run_solution(task: Task, solution: str | None = None, timeout: float = 5.0) -> bool:
    """True iff the solution drives the task's verifier to pass (subprocess-isolated)."""
    return run_capturing(task, solution, timeout)[0]


def validate_task(task: Task, timeout: float = 5.0) -> bool:
    """A task is valid only if its own gold solution satisfies its verifier."""
    return run_solution(task, solution=None, timeout=timeout)
