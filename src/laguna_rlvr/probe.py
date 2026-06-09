"""Probe phase: eval a candidate env × model via `prime eval run`, normalize rollouts to per-task records."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def build_eval_command(env: str, model: str, provider: str, num_examples: int,
                       rollouts_per_example: int, max_tokens: int, temperature: float,
                       output_dir: Path, env_args: dict | None = None,
                       api_base_url: str | None = None, api_key_var: str | None = None) -> list[str]:
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
    if api_base_url:                       # e.g. local Ollama at http://localhost:11434/v1
        cmd += ["--api-base-url", api_base_url]
    if api_key_var:
        cmd += ["--api-key-var", api_key_var]
    return cmd


def _pick(r: dict, key: str):
    """Read a rollout field, tolerating verifiers' `_`-prefixed rubric-metric keys and the metrics dict."""
    metrics = r.get("metrics") or {}
    for k in (key, f"_{key}"):
        if k in r:
            return r[k]
    return metrics.get(key, metrics.get(f"_{key}"))


def normalize_records(raw: list[dict], success_key: str, reward_key: str) -> list[dict]:
    """Project raw eval rollouts onto the {success, reward} schema the report consumes."""
    return [{"success": bool(_pick(r, success_key)), "reward": float(_pick(r, reward_key))} for r in raw]


def _load_raw_results(output_dir: Path) -> list[dict]:
    files = sorted(output_dir.rglob("results.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no results.jsonl under {output_dir}")
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
                             cfg.model.max_tokens, cfg.model.temperature, raw_dir, env_args,
                             api_base_url=cfg.model.get("api_base_url"),
                             api_key_var=cfg.model.get("api_key_var"))
    subprocess.run(cmd, check=True)

    records = normalize_records(_load_raw_results(raw_dir), cfg.probe.success_key, cfg.probe.reward_key)
    out = probe_dir / f"{cfg.env.module}__{cfg.model.slug}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in records))
    print(f"wrote {len(records)} records → {out}")


if __name__ == "__main__":
    main()
