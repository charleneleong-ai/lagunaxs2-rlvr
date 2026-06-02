# Multi-scaffold tool-use — design

## Why
A model's agentic score is `base × harness`. Laguna XS.2 ships co-designed with the **Pool/ACP**
harness and was RL-trained on its tool-call format, so the *surface syntax* of a tool call is part of
what it learned. Training/eval in one fixed syntax overfits the policy to that harness. MiniMax-M2's
fix is to **sample under multiple scaffolds so the policy generalizes beyond any single tool layout**.

## The evidence (why this isn't hypothetical)
`ocr_tool` on hosted Laguna XS.2 (Prime inference), same 4 docs, tool-call **syntax the only change**:

| scaffold | success | note |
| --- | --- | --- |
| `line` (`ocr: invoice.png`) | **4/4** | |
| `json` (`{"tool":"ocr","image_id":...}`) | **4/4** | |
| `poolside` (native `<tool_call>ocr<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`) | **4/4** | |
| `xml` (Hermes `<tool_call>{json}</tool_call>`) | **0/4** | model reverts to its native dialect |
| `mixed` (round-robin) | **3/4**, reward `[1.1,0,1.1,1.1]`, std 0.48 | learnable variance |

The `xml` collapse is the finding: prompted for Hermes-JSON-in-tags, Laguna **ignored the instruction
and emitted its own `poolside_v1` dialect** (`<tool_call>name<arg_key>…<arg_value>…`), which the Hermes
parser can't read → 0%. Tool-call syntax alone swung one model 100% → 0%. (Native format from the HF
card: OpenAI structured tool-calls via vLLM `--tool-call-parser poolside_v1`, interleaved reasoning,
temp 0.7 / top_k 20.)

## Decision
A shared [`scaffold`](../../src/laguna_rlvr/scaffold.py) module renders tool instructions and parses a
tool call in **4 surface syntaxes** — `line`, `xml`, `json`, `poolside` — selected per task; `mixed`
round-robins them. An env threads the chosen `fmt` through `info` and calls `render_instructions` /
`parse_call`; nothing else changes (no inference-call wiring needed for these text formats).

- **Scope:** single-string-arg tool calls (the tool-mediated family — `ocr_tool`, and `frontend_design`
  next). Multi-arg executed-code tools (`general_agent`) are a different paradigm and a non-goal.
- **`poolside`** matches the text Laguna actually emits in message *content* (confirmed by eval),
  distinct from the structured `tool_calls` field a `poolside_v1` server exposes — a future "native"
  scaffold would advertise vf tool schemas and read that field.

## Payoff
`mixed` turns harness-brittleness into **reward variance** — the learnable signal an RLVR run uses to
*teach* Laguna the syntaxes it doesn't know natively (Hermes) while reinforcing the ones it does. The
result is a policy robust to harness change instead of tuned to one tool layout.

## Native scaffold — works on the Prime endpoint (no Harbor needed)
`scaffold="native"` is the real Pool/ACP path: the env advertises tool schemas (`to_tool_defs` →
verifiers `tool_defs`, **vf.Tool format** — *not* the legacy OpenAI `{type,function}` wrapper, which
verifiers rejects) so the model emits structured `tool_calls`, read by `parse_native` instead of text.
Env-level mode (not round-robined into `mixed`, since it needs schema advertisement).

**Validated end-to-end on hosted Laguna XS.2 (Prime inference): 4/4 (reward 1.1, 3 turns).** The Prime
endpoint already supports function-calling — `tool_defs` advertised, Laguna returned a real structured
call. The one gotcha: verifiers returns tool_calls in a **flat JSON-string shape** (`{"name","arguments"}`
at top level, the call a JSON string), *not* the nested OpenAI `{"function":{...}}` form — `parse_native`
tolerates both. So no separate vLLM `poolside_v1` / Harbor endpoint is required.

Full Laguna table now: `line` 4/4 · `json` 4/4 · `poolside`(text) 4/4 · **`native`(structured) 4/4** ·
`xml`(Hermes) 0/4 · `mixed` 3/4.

## Next
- ✅ `frontend_design` wired; ✅ `native` works on Prime; ✅ dead env stubs removed.
- RL `ocr_tool`/`frontend_design` with `scaffold="mixed"` on the free Laguna slot.
