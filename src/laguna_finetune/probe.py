"""Probe phase: eval a candidate env × model via `prime eval run`, normalize rollouts to per-task records."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def build_eval_command(env: str, model: str, provider: str, num_examples: int,
                       rollouts_per_example: int, max_tokens: int, temperature: float,
                       output_dir: Path, env_args: dict | None = None) -> list[str]:
    cmd = [
        "prime", "eval", "run", env,
        "--provider", provider,
        "--model", model,
        "--num-examples", str(num_examples),
        "--rollouts-per-example", str(rollouts_per_example),
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--save-results", "--output-dir", str(output_dir),
    ]
    if env_args:
        cmd += ["--env-args", json.dumps(env_args)]
    return cmd


def normalize_records(raw: list[dict], success_key: str, reward_key: str) -> list[dict]:
    """Project raw eval rollouts onto the {success, reward} schema the report consumes."""
    return [{"success": bool(r[success_key]), "reward": float(r[reward_key])} for r in raw]


def _load_raw_results(output_dir: Path) -> list[dict]:
    files = sorted(output_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no eval results under {output_dir}")
    return [json.loads(line) for line in files[-1].read_text().splitlines() if line.strip()]


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    probe_dir = Path(cfg.paths.results) / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = probe_dir / f"_raw_{cfg.env.module}__{cfg.model.slug}"

    env_args = {**OmegaConf.to_container(cfg.env.args, resolve=True),
                **OmegaConf.to_container(cfg.reward, resolve=True)}  # reward group flows into the env
    cmd = build_eval_command(cfg.env.module, cfg.model.name, cfg.model.provider,
                             cfg.probe.num_examples, cfg.probe.rollouts_per_example,
                             cfg.model.max_tokens, cfg.model.temperature, raw_dir, env_args)
    subprocess.run(cmd, check=True)

    records = normalize_records(_load_raw_results(raw_dir), cfg.probe.success_key, cfg.probe.reward_key)
    out = probe_dir / f"{cfg.env.module}__{cfg.model.slug}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in records))
    print(f"wrote {len(records)} records → {out}")


if __name__ == "__main__":
    main()
