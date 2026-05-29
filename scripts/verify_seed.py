"""Verify an Adaption seed/dataset JSONL is RL-ready.

For each row: the reference_solution must pass ALL its tests, and an empty stub must fail at
least one (discriminating). Exits non-zero if any row is broken — wire it into CI / pre-adapt.

Usage: uv run python scripts/verify_seed.py docs/adaption/seed_multilingual_coding.jsonl
"""
from __future__ import annotations

import json
import sys

from laguna_rlvr.code_exec import score_code


def verify(path: str) -> bool:
    all_ok = True
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        program = f"{r.get('setup', '')}\n{r['reference_solution']}" if r.get("setup") else r["reference_solution"]
        passed, total = score_code(program, r["tests"])
        empty, _ = score_code("", r["tests"])
        good = passed == total and empty == 0
        all_ok &= good
        print(f"{'OK ' if good else 'BAD'} [{r.get('language', '?')}/{r.get('difficulty', '?')}] "
              f"ref {passed}/{total} | empty-stub {empty}/{total}")
    print("ALL VERIFIABLE" if all_ok else "SOME ROWS BROKEN")
    return all_ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/adaption/seed_multilingual_coding.jsonl"
    sys.exit(0 if verify(path) else 1)
