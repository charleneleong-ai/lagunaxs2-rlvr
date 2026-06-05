"""Per-task qa accuracy breakdown for specific runs by name substring.

Usage: wandb_detail.py <entity/project> <name_substr> [<name_substr> ...]
"""

import sys

import wandb

entity_project = sys.argv[1]
substrs = sys.argv[2:]
api = wandb.Api()

for r in api.runs(entity_project, order="-created_at"):
    if not any(sub in r.name for sub in substrs):
        continue
    s = r.summary
    accs = {k: s[k] for k in s.keys() if k.startswith(("qa/metrics/acc", "read/metrics/acc")) and isinstance(s.get(k), (int, float))}
    print(f"\n## {r.name}  [{r.state}]  best={s.get('qa/metrics/best')}")
    for k in sorted(accs, key=lambda k: -accs[k]):
        print(f"   {accs[k]:.4f}  {k.split('acc_')[-1]}")
