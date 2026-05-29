# agentic-repair

Composite-reward, long-horizon agentic **code repair**. The agent is given a *buggy* function
(a one-edit mutation injected into an MBPP reference solution) and makes a surgical fix over
multiple turns. The reward is a reusable, benchmark-agnostic weighted rubric:

| Component | Reward | Idea |
|---|---|---|
| `correctness` | hidden-test pass fraction | base |
| `efficiency` | solve in fewer turns | #1 |
| `minimal_diff` | stay close to the buggy original (surgical edit) | #3 |
| `self_verify` | model's own tests agree with hidden outcome (opt-in) | #2 |

Tier 1 = injected-bug MBPP (local, $0). Tier 2 = point the same rubric at a real SWE env
(`mini-swe-agent-plus`) for the long-horizon result.

```bash
export OLLAMA_API_KEY=ollama
uv run python -m laguna_rlvr.probe env=agentic_repair model=ollama   # dev
prime eval run agentic_repair -m poolside/laguna-xs.2                     # Laguna
```
