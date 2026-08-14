# Why our tool loop wastes turns

Type: research
Status: resolved
Blocked by:

## Question

How do Waku and similar agents get reliable tool calls? What should replace our "reply with ONLY JSON" + last-8-turns transcript so RESEARCH writes cards and CODE changes the kernel?

## Answer

Live 2026-08-14: eval `115/166` research tools were `invalid_json` (`memory/daily/eval_report.json`). `write_card` count is 0. `harvest_cards` fired 2 times all day. CODE `write_brief` 11, `write_methods` 2, then `apply_recipe` always.

### 1. Why `invalid_json` happens

`StageAgent._next_action` (`src/kaggle_agent/agents/loop.py`) asks the model for **text JSON** `{"tool","args"}` with `max_tokens=400`. `parse_tool_call` only accepts a JSON object (optional ``` fence, else first `{`…last `}`). Anything else becomes tool name `invalid_json` → `unknown tool invalid_json`.

`ZenClient.chat` (`src/kaggle_agent/llm/zen_client.py`) never sends `tools` / `tool_choice`. It returns `message.content`, or if empty, `reasoning_content`. It **drops** `message.tool_calls`.

Official DeepSeek v4 (`https://api-docs.deepseek.com/`):

- Thinking is **on by default**, effort `high`. CoT goes to `reasoning_content`. `max_tokens` counts those tokens.
- Native tools are `tools: [{type:function,...}]`. The call is in `choices[0].message.tool_calls`, `finish_reason=tool_calls`, and `content` is often `""`.
- `response_format: {type: json_object}` is a separate path. We do not send it.

Trace `memory/daily/traces/2026-08-14.jsonl` (cycle `20260814-162335`): almost every LLM row is `tokens_out: 400` (hard cap) with `chars` 1600–1800, or 6–55. That is truncated English CoT, not `{"tool":"harvest_cards","args":{}}`. Parser fails. Next user blob is `tool=invalid_json … unknown tool`. The model never sees a schema, only a name list in the system string.

`write_card` needs `markdown` in `args`. After 400 thinking tokens, that payload cannot fit. The model also `done`s early; `accept_done` rejects (`cards_feasible`) and the 8-turn window then holds only reject + invalid lines.

### 2. What Waku does instead

Yes: **native tools**. `waku/loop/agent.py` `run_loop` calls `client.messages.create(..., tools=tools.schemas(), max_tokens=2048)`. Stop = no `tool_use` blocks, or iteration cap. Results go back as `tool_result` / OpenAI `role: tool`.

`waku/loop/models.py`: DeepSeek is an **OpenAI-wire** provider (`https://api.deepseek.com`). `OpenAICompatClient` maps Anthropic `tools` → `tools[].function`, and `tool_calls[].function.arguments` → `tool_use`. It does **not** parse JSON out of `content`.

Waku default `max_tokens` is 2048, not 400. History is real messages, not a sliced string transcript.

DeepSeek docs (`/guides/tool_calls`, `/guides/thinking_mode#tool-calls`): pass `tools`; execute; append the **full assistant message** (`content` + `reasoning_content` + `tool_calls`) plus `role: tool` / `tool_call_id`. Missing `reasoning_content` after a tool turn → HTTP 400.

### 3. Smallest change so harvest/write happens every cycle

Do not wait for a full Waku port.

**Smallest (one cycle guarantee):** in `Orchestrator._research` (`orchestrator.py`), after snapshot + browser, **always call `harvest_cards` once** (it already runs `_source_cards` + `write_methods_sidecar`). Emit a `tool` trace so evals `research_wrote_card` pass. The LLM loop can still search / `write_card` after.

**Do not invert `no_zen_sequence` as the live path.** It only runs when `zen is None` (`loop.py` 128–133). With a live key it is dead. Seeding `harvest_cards` in the orchestrator is clearer than overloading that list.

**Real loop fix (next slice, not smallest):** `ZenClient.chat` accepts `tools=` and returns `(content, tool_calls)`. `StageAgent` uses that. For DeepSeek tool turns, set `thinking: disabled` **or** `max_tokens>=2048` plus `tool_choice: required` until `done`. Keep `parse_tool_call` only as a fallback.

`write_card` stays optional. `harvest_cards` is the reliable card writer.

### 4. How CODE should apply cards (not just `write_brief`)

Today (`orchestrator.py` `_code`, `agents/code.py`, `competitions/rsna_knee/pipeline/recipe.py`):

1. Agent may `write_brief`. `accept_done` is only `brief_path.is_file()`.
2. `no_zen_sequence=["write_brief"]` writes a generic line if there is no zen.
3. After the loop, **always** `apply_recipe` (fit metadata ranker → `weights.json`).
4. `apply_from_cards` only writes `pipeline/methods_applied.md`. It does not attach datasets, edit the notebook, or change inference.

`CODE_SYSTEM` says call `done` after a brief exists. `write_methods` is optional. Cards never reach the kernel.

What CODE should do:

- `accept_done` = valid `methods.json` (`methods_payload_ok`) **and** a brief. Reject `done` until `write_methods` (or harvest sidecar) is good.
- After the loop: **`apply_from_cards` first**. Use `dataset_sources` / `model_sources` / `infer_hints` / `implement_steps` to pin kernel attachments and ID discovery. Keep `apply_recipe` as the local ranker **only**, not the whole experiment.
- Kernel / `notebook_builder` must read `methods.json`, not `code_brief.md` prose.

### 5. What to delete or simplify

| Thing | Verdict |
|--------|---------|
| `no_zen_sequence` | Delete as a live substitute. Tests can call tools directly. If no zen, fail the stage or run one deterministic harvest/write_plan — do not pretend it is an agent. |
| Last-8 transcript (`transcript[-8:]`) | Drops the context pack after a few `invalid_json` rows. Keep turn 0 (pack) + last N observations, or a real `messages[]` like Waku. |
| `max_tokens=400` | Hits every live call (`tokens_out: 400`). Too small for default thinking. Tool turns: disable thinking **or** ≥2048. `write_card` markdown cannot live in 400 tokens. |
| JSON-in-content + `Reply with ONLY valid JSON` | Replace with native `tools`. Optional `json_object` only if we stay on text JSON. |
| CODE `done` after brief only | Tighten gate as in §4. |

Cite: `agents/loop.py`, `research/agent.py`, `agents/plan.py`, `agents/code.py`, `orchestrator.py` `_research`/`_code`, `llm/zen_client.py`, `research/source_cards.py` `write_methods_sidecar`/`cards_feasible`, Waku `waku/loop/agent.py` + `models.py`, DeepSeek `/guides/tool_calls` and `/api/create-chat-completion` (`tools`, `tool_choice`, `tool_calls`).
