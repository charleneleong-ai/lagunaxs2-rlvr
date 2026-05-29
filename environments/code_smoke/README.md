# code-smoke

Multi-turn, no-Docker code-repair env: the agent writes a Python function, the env runs its
assert-tests in a subprocess, returns the failing ones as feedback, and the agent iterates until
they pass or `max_turns` is hit. Reward = test-pass fraction + an efficiency nudge for solving in
fewer turns (`reward=binary` for pass/fail only).

Dev-friendly: runs locally with any OpenAI-compatible model (e.g. Ollama), $0, no sandbox — used
to validate the probe → reward → report pipeline before the real targets.

```bash
export OLLAMA_API_KEY=ollama
uv run python -m laguna_finetune.probe env=code_smoke model=ollama
```
