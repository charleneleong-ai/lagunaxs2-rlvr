"""Per-language SWE env targeting poolside's reported multilingual headroom (57.7% vs 69.9% Verified).

Deferred stub — build this after the terminal_bench_curated env, which is the working template:
mirror its structure (a verifiers env + a shaped partial-credit reward over per-test results),
swapping the Terminal-Bench harness for a per-language SWE taskset.
"""
from __future__ import annotations

import verifiers as vf


def load_environment(langs: list[str] | None = None, max_turns: int = 50,
                     fn: str = "shaped", efficiency_weight: float = 0.1, **kwargs) -> vf.Environment:
    raise NotImplementedError(
        f"swe_multilingual not built yet. Mirror terminal_bench_curated for langs="
        f"{langs or ['go', 'rust', 'typescript']}: wrap a per-language SWE taskset and reuse the "
        "shaped partial-credit reward. Probe Laguna's per-language base rate to pick the target.")
