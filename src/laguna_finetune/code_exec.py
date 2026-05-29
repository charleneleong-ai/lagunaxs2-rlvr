"""Run model-generated code against assert-style tests in a subprocess. No Docker.

Each test runs in its own `python -c` subprocess so one failure/timeout doesn't sink the rest,
giving a per-test pass count for a dense partial-credit reward.
"""
from __future__ import annotations

import re
import subprocess
import sys

_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.DOTALL)  # any/no lang label, case-insensitive


def extract_code(text: str) -> str:
    """Pull the first fenced code block; fall back to the whole text."""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def score_code(code: str, tests: list[str], timeout: float = 5.0) -> tuple[int, int]:
    """Return (passed, total): how many assert-tests the code satisfies."""
    passed = 0
    for test in tests:
        program = f"{code}\n{test}\n"
        try:
            result = subprocess.run([sys.executable, "-c", program],
                                    capture_output=True, timeout=timeout)
            passed += result.returncode == 0
        except subprocess.TimeoutExpired:
            pass
    return passed, len(tests)
