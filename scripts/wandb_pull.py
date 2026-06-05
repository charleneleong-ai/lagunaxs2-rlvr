"""Pull recent runs from a W&B project and print a compact one-line-per-run summary.

Usage: wandb_pull.py <entity/project> [days]
Reads WANDB_API_KEY from the environment (source .env first).
"""

import sys
from datetime import datetime, timedelta, timezone

import wandb

entity_project = sys.argv[1] if len(sys.argv) > 1 else "chaleong/laguna-mm-adapter"
days = int(sys.argv[2]) if len(sys.argv) > 2 else 6
cutoff = datetime.now(timezone.utc) - timedelta(days=days)

api = wandb.Api()
runs = api.runs(entity_project, order="-created_at")


def g(s, *keys):
    for k in keys:
        v = s.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


print(f"# {entity_project} — last {days}d\n")
print(f"{'created':16} {'state':9} {'qa_best':>8} {'qa_acc':>7} {'code_valid':>10}  run")
for r in runs:
    created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00")) if isinstance(r.created_at, str) else r.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if created < cutoff:
        continue
    s = r.summary
    qa_best = g(s, "qa/metrics/best", "qa_accuracy")
    qa_acc = g(s, "qa/metrics/accuracy", "qa_accuracy")
    cv = g(s, "val/metrics/code_valid")
    fb = lambda x: f"{x:.4f}" if x is not None else "   -  "
    print(f"{created:%Y-%m-%d %H:%M}  {r.state:9} {fb(qa_best):>8} {fb(qa_acc):>7} {fb(cv):>10}  {r.name}")
