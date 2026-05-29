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


def message_text(message) -> str:
    """Text of a chat message — handles a str, a dict, or a verifiers pydantic Message (.content)."""
    if isinstance(message, str):
        return message
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal content parts
        return "".join(p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") for p in content)
    return ""


def run_python(program: str, timeout: float = 5.0) -> subprocess.CompletedProcess | None:
    """Run a Python program in an isolated subprocess; return the result, or None on timeout."""
    try:
        return subprocess.run([sys.executable, "-c", program],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def score_code(code: str, tests: list[str], timeout: float = 5.0) -> tuple[int, int]:
    """Return (passed, total): how many assert-tests the code satisfies."""
    passed = 0
    for test in tests:
        r = run_python(f"{code}\n{test}\n", timeout)
        passed += r is not None and r.returncode == 0
    return passed, len(tests)
