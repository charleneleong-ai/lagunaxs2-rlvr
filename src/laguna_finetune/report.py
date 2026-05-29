"""Report phase: rank probe domains by base_rate × reward-variance (the learnable-signal metric)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@dataclass(frozen=True)
class DomainRanking:
    env: str
    model: str
    n: int
    base_rate: float
    variance: float
    signal: float
    learnable: bool


def load_records(probe_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for f in sorted(probe_dir.glob("*.jsonl")):
        env, _, model = f.stem.partition("__")
        for line in f.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rows.append({"env": env, "model": model,
                             "success": bool(rec["success"]), "reward": float(rec["reward"])})
    return pd.DataFrame(rows, columns=["env", "model", "success", "reward"])


def rank(df: pd.DataFrame) -> list[DomainRanking]:
    """Rank (env, model) groups by signal = base_rate × reward-variance, descending."""
    rankings: list[DomainRanking] = []
    for (env, model), g in df.groupby(["env", "model"], sort=False):
        base = float(g["success"].mean())
        variance = float(g["reward"].var(ddof=0)) if len(g) > 1 else 0.0
        rankings.append(DomainRanking(env, model, len(g), base, variance, base * variance, variance > 0.0))
    return sorted(rankings, key=lambda r: r.signal, reverse=True)


def render_markdown(rankings: list[DomainRanking]) -> str:
    header = ("| env | model | n | base_rate | variance | signal | learnable |\n"
              "|---|---|---|---|---|---|---|\n")
    body = "\n".join(
        f"| {r.env} | {r.model} | {r.n} | {r.base_rate:.3f} | {r.variance:.4f} | "
        f"{r.signal:.4f} | {'✅' if r.learnable else '⚠️ flat'} |"
        for r in rankings)
    return header + body + "\n"


def _plot(rankings: list[DomainRanking], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r.env}\n{r.model}" for r in rankings]
    colors = ["#2a9d8f" if r.learnable else "#e76f51" for r in rankings]
    fig, ax = plt.subplots(figsize=(max(4.0, len(rankings) * 1.6), 4.0))
    ax.bar(labels, [r.signal for r in rankings], color=colors)
    ax.set_ylabel("learnable signal (base_rate × variance)")
    ax.set_title("Probe — headroom ranking")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    probe_dir = Path(cfg.paths.results) / "probe"
    df = load_records(probe_dir)
    if df.empty:
        print(f"no probe records under {probe_dir} — run the probe first")
        return
    rankings = rank(df)
    markdown = render_markdown(rankings)
    (probe_dir / "ranking.md").write_text(markdown)
    _plot(rankings, probe_dir / "ranking.png")
    print(markdown)


if __name__ == "__main__":
    main()
