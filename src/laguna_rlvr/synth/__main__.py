"""Run the synthesizer against an OpenAI-compatible endpoint → a verifiable JSONL corpus.

  SYNTH_BASE_URL  (default http://localhost:11434/v1 — local Ollama)
  SYNTH_MODEL     (default qwen3:8b; use poolside/laguna-xs.2 via Prime for stronger tasks)
  SYNTH_API_KEY   (default "ollama"; your Prime/OpenAI key for hosted models)

Example:
  SYNTH_MODEL=qwen3:8b uv run python -m laguna_rlvr.synth library warehouse clinic
Only self-consistent tasks (gold satisfies verifier) are written — invalid generations are dropped.
"""
from __future__ import annotations

import json
import os
import sys

import httpx

from laguna_rlvr.synth.generate import build_corpus


def _chat(base_url: str, api_key: str, model: str):
    def call(prompt: str) -> str:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 1500},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    return call


def main() -> None:
    base = os.environ.get("SYNTH_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("SYNTH_MODEL", "qwen3:8b")
    key = os.environ.get("SYNTH_API_KEY", "ollama")
    domains = sys.argv[1:] or ["library", "warehouse", "clinic"]
    out = "docs/adaption/general_agent_corpus.jsonl"

    corpus = build_corpus(_chat(base, key, model), domains, tiers=2)
    with open(out, "w") as f:
        f.write("\n".join(json.dumps(t.to_dict()) for t in corpus))
    print(f"synthesized {len(corpus)} self-consistent task(s) across {len(domains)} domain(s) → {out}")


if __name__ == "__main__":
    main()
