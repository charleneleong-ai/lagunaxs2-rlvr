# general-agent

Multi-turn **tool-use** env over general-agent-style tasks (à la Prime Intellect's general-agent
corpus): each task = a Pydantic DB schema + tool APIs + NL instruction + gold tool-call sequence +
verification fn, tagged with a difficulty tier.

The agent is shown the instruction + tool signatures and emits a ```python block of tool calls each
turn; the env replays the accumulated calls in a subprocess, returns captured stdout as the
observation, and scores with the task's verifier. Reward = solved + an efficiency nudge (fewer turns).
Complements the code-repair envs by exercising **interleaved tool-calling**. $0, no sandbox.

Tasks load from `source="builtin"` (a small tiered day-spa set) or a JSONL corpus (one `Task` dict
per line — the synthesizer's output target).

```bash
export OLLAMA_API_KEY=ollama
uv run python -m laguna_rlvr.probe env=general_agent model=ollama
prime eval run general_agent -m poolside/laguna-xs.2
```
