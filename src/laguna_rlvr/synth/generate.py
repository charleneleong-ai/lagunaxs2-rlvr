"""Task synthesizer: grow a verifiable general-agent corpus with an LLM, gated by self-consistency.

The loop (the RLVR self-evolving engine): an LLM proposes a domain task (schema + tools + instruction
+ gold + verifier); we KEEP it only if its gold solution drives its verifier to pass (`validate_task`,
executed in a subprocess). Valid tasks are then *evolved* to higher tiers via the corpus's evolution
strategies. Difficulty is calibrated against a gating solver — keep a tier only if its pass-rate sits
in the learnable band (`report.rank`'s success-spread sweet spot), so the curriculum tracks the model.

`call_llm: prompt -> completion` is injected, so this is provider-agnostic (Ollama / Prime inference /
OpenAI) and unit-testable with a fake. No keys, no network in the pure parsing/validation path.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from laguna_rlvr.synth.task import Task, run_solution, validate_task

# The 9 evolution strategies from the general-agent corpus (t0 -> t4).
EVOLUTION_STRATEGIES = [
    "multi_step_reasoning", "conditional_rules", "cross_entity_coupling", "stricter_thresholds",
    "larger_db", "schema_extension", "tool_proliferation", "noisy_instructions", "ambiguity_resolution",
]

_FORMAT = (
    "Output EXACTLY these sections:\n"
    "DOMAIN: <short_snake_case_domain>\nTIER: <0-4>\nINSTRUCTION: <one natural-language goal>\n"
    "[SCHEMA]\n```python\n<Pydantic BaseModel entity classes>\n```\n"
    "[TOOLS]\n```python\n<a module-level `db` dict/state + tool functions that mutate it>\n```\n"
    "[GOLD]\n```python\n<tool-call statements that accomplish the goal>\n```\n"
    "[VERIFIER]\n```python\ndef verify() -> bool:\n    <read `db`, return True iff the goal is met>\n```\n"
    "Constraints: pure stdlib + pydantic; the gold solution MUST make verify() return True; "
    "tools must be deterministic; no I/O or network."
)

SYNTH_PROMPT = (
    "You design verifiable tool-use tasks for an agent benchmark.\n\n"
    "Create ONE self-contained task at difficulty tier {tier}{hint}.\n\n" + _FORMAT
)

EVOLVE_PROMPT = (
    "Here is a tool-use task:\n\nDOMAIN: {domain}\nINSTRUCTION: {instruction}\n"
    "[SCHEMA]\n```python\n{schema}\n```\n[TOOLS]\n```python\n{tools}\n```\n"
    "[GOLD]\n```python\n{gold}\n```\n[VERIFIER]\n```python\n{verifier}\n```\n\n"
    "Make it harder using the '{strategy}' strategy (raise the tier by 1). Keep it self-consistent "
    "(the new gold must satisfy the new verifier).\n\n" + _FORMAT
)

_SECTION = re.compile(r"\[(SCHEMA|TOOLS|GOLD|VERIFIER)\]\s*```(?:python)?\n(.*?)```", re.DOTALL)


def parse_task(text: str) -> Task | None:
    """Parse a synthesizer completion into a Task, or None if any section is missing."""
    domain = re.search(r"DOMAIN:\s*(.+)", text)
    instruction = re.search(r"INSTRUCTION:\s*(.+?)(?:\n\[|\nTIER:|\Z)", text, re.DOTALL)
    tier = re.search(r"TIER:\s*(\d)", text)
    sections = {m.group(1): m.group(2).strip() for m in _SECTION.finditer(text)}
    if not (domain and instruction) or not {"SCHEMA", "TOOLS", "GOLD", "VERIFIER"} <= sections.keys():
        return None
    return Task(domain=domain.group(1).strip(), tier=int(tier.group(1)) if tier else 0,
                schema_code=sections["SCHEMA"], tools_code=sections["TOOLS"],
                instruction=instruction.group(1).strip(), gold_solution=sections["GOLD"],
                verifier_code=sections["VERIFIER"])


def synthesize(call_llm: Callable[[str], str], domain_hint: str = "", tier: int = 0,
               max_retries: int = 3, timeout: float = 5.0) -> Task | None:
    """Generate one self-consistent task (gold satisfies verifier), retrying on parse/validate failure."""
    hint = f" for the '{domain_hint}' domain" if domain_hint else ""
    for _ in range(max_retries):
        task = parse_task(call_llm(SYNTH_PROMPT.format(tier=tier, hint=hint)))
        if task and validate_task(task, timeout):
            return task
    return None


def evolve(call_llm: Callable[[str], str], task: Task, strategy: str,
           max_retries: int = 3, timeout: float = 5.0) -> Task | None:
    """Harden a valid task one tier via an evolution strategy; keep only if still self-consistent."""
    prompt = EVOLVE_PROMPT.format(domain=task.domain, instruction=task.instruction,
                                  schema=task.schema_code, tools=task.tools_code,
                                  gold=task.gold_solution, verifier=task.verifier_code, strategy=strategy)
    for _ in range(max_retries):
        evolved = parse_task(call_llm(prompt))
        if evolved and validate_task(evolved, timeout):
            return evolved
    return None


def is_learnable(task: Task, solver: Callable[[Task], bool], samples: int = 4) -> bool:
    """Tier gating: a task is in the learnable band iff the solver sometimes-but-not-always solves it
    (0 < pass_rate < 1) — the success-spread sweet spot that yields an RL gradient."""
    passes = sum(bool(solver(task)) for _ in range(samples))
    return 0 < passes < samples


def build_corpus(call_llm: Callable[[str], str], domains: list[str], tiers: int = 3,
                 timeout: float = 5.0) -> list[Task]:
    """Synthesize a tier-0 task per domain, then evolve each up `tiers` levels — all self-consistent."""
    corpus: list[Task] = []
    for i, domain in enumerate(domains):
        task = synthesize(call_llm, domain_hint=domain, tier=0, timeout=timeout)
        if task is None:
            continue
        corpus.append(task)
        for t in range(1, tiers):
            task = evolve(call_llm, task, EVOLUTION_STRATEGIES[t % len(EVOLUTION_STRATEGIES)], timeout=timeout)
            if task is None:
                break
            corpus.append(task)
    return corpus
