"""Dry-run a generalized adapter config against its VRAM budget."""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from laguna_rlvr.mm_adapter import plan_from_config, render_plan, validate_gpu_budget


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a configs/mm_adapter/*.toml file")
    args = parser.parse_args(argv)

    with args.config.open("rb") as fh:
        plan = plan_from_config(tomllib.load(fh))
    print(render_plan(plan))
    return 1 if validate_gpu_budget(plan) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
