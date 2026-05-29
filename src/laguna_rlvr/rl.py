"""RL phase: launch Prime training on the chosen env, gated by a learnable probe signal."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from laguna_rlvr.report import DomainRanking, load_records, rank


def should_train(rankings: list[DomainRanking], env: str) -> tuple[bool, str]:
    """Gate the one Laguna run: refuse domains with no probe data or zero reward variance."""
    match = next((r for r in rankings if r.env == env), None)
    if match is None:
        return False, f"no probe data for env '{env}' — run the probe first"
    if not match.learnable:
        return False, (f"env '{env}' has zero reward variance — RL has no gradient; "
                       "reshape the reward or pick another domain")
    return True, f"env '{env}' learnable (base_rate={match.base_rate:.3f}, variance={match.variance:.4f})"


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    probe_dir = Path(cfg.paths.results) / "probe"
    rankings = rank(load_records(probe_dir)) if probe_dir.exists() else []
    ok, reason = should_train(rankings, cfg.env.module)
    print(reason)
    if not ok:
        raise SystemExit(1)
    # TODO(event): launch `prime train <cfg.rl.config>` once the prime-rl config is authored.
    print(f"would launch: prime train {cfg.rl.config} on {cfg.model.name}")


if __name__ == "__main__":
    main()
