# Review: deep research stage (2026-08-15)

Audit of the deep-research stage and its neighbors (research agent tool-loop, browser research, source cards). Evidence: repo code + daily logs + `memory/research-deep/` artifacts. No code was changed.

Scope: three questions — (1) does the stage do real research, (2) search provider / tool-call parsing / data flow, (3) browser-use vs browser-harness usage.

## Q1 — Does deep research collect real information? Verdict: PARTIAL

The stage is not a shell. It runs real searches against four sources and pulls real content, and learnings/sources vary per run. But the generic-web results are mostly irrelevant, and the final report is largely LLM-synthesized from the competition snapshot + model knowledge, not from fetched pages.

Evidence it does real research:

- KaggleSource.search calls the Kaggle API `kernels_list` and `content` calls `kernels_pull` to cache notebooks: `src/kaggle_agent/research/deep.py:69-127`. The pulled artifacts are real: 16 notebooks in `competitions/rsna_knee/research-cache/*.ipynb` with genuine source (verified `rsna-knee-read-the-report-then-the-knee.ipynb`).
- ArxivSource queries `export.arxiv.org/api/query` (`deep.py:178-213`); GithubSource queries `api.github.com/search/repositories` and fetches raw READMEs (`deep.py:224-264`); WebSource scrapes DuckDuckGo HTML SERP (`deep.py:275-306`).
- Fetched content is fed to the LLM and distilled into learnings: `_fetch_all` (`deep.py:455-476`), `_distill` builds `<content>` blocks from fetched text (`deep.py:478-498`). Learnings and visited URLs propagate: `deep.py:541-542, 610`.
- The report carries real source URLs: `_write_report` appends `## Sources` from `visited` (`deep.py:566-575`). `deep-20260814-060200.md:296-318` lists 23 real arXiv IDs; `deep-20260814-090859.md:405-431` lists arXiv plus `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview` and `https://www.kaggle.com/code/dorrech1/kneexxx` — real hits that flowed from DDG search.
- Daily logs confirm live runs with real counts: `deep depth=2 queries=3 total=1 … deep research learnings=16 sources=23 queries=4 error=none` (`memory/daily/2026-08-14.md:15-19`).

Evidence it is only partially research:

- Both surviving reports openly state the fetched snippets were irrelevant and that competition facts came from the injected snapshot, not from the web: "The supplied browser/search learnings were mostly negative" (`deep-20260814-090859.md:37-47`) and "the reliable facts below come primarily from the Kaggle API/current-knowledge block and the prompt itself" (`deep-20260814-090859.md:47`); same in `deep-20260814-060200.md:261-275`. The competition-specific detail in those reports (kernel list, votes, scores) matches the snapshot passed into `_deep_prompt` (`orchestrator.py:741-745` reads `memory/research.md[:4000]`), i.e. the LLM synthesized it, with the fetched pages contributing little.
- The `## Deep research digest` section never survives to the file PLAN reads: `memory/research.md` currently has `## Browser (read-only)` and `## Method cards` sections but no digest (grep "Deep research digest" → no match). The snapshot overwrite at the start of each RESEARCH phase (`apply_snapshot.py:22`) plus failed/late deep runs erase or prevent it.

## Q2 — Search provider, tool calls, JSON parsing, data flow. Verdict: PARTIAL

Search provider: there is no dedicated web_search API. All research LLM calls go to DeepSeek official via `ZenClient` (`build_llm_client` builds a plain `ZenClient` to `https://api.deepseek.com` with `DEEPSEEK_API_KEY`, `llm/fallback.py:124-135`; `config/settings.yaml:11-22`); the "web search" is a DuckDuckGo HTML scrape (`deep.py:275-306`) plus arXiv/GitHub/Kaggle APIs.

Tool-call parsing works mechanically but fails constantly in practice:

- `parse_tool_call` strips fences, tries JSON, falls back to first `{...}` span (`agents/loop.py:26-47`); native tool_calls are parsed in `zen_client.py:17-37` and preferred (`loop.py:195-197`); a zero-temperature retry runs once (`loop.py:201-219`).
- Evidence of failure: 286 `invalid_json` turn lines in `memory/daily/2026-08-14.md` vs 64 real `search`/`fetch_url`/`pull_kernel` turns. Worst run 09:34-09:39: 40 turns, ~35 `invalid_json`, `stop=turn_cap`, and deep research was never invoked (`2026-08-14.md:138-179`). The free/`deepseek-v4-flash` model returns prose, not the requested `{"tool": ...}` JSON, most of the time.
- The deep stage's strict-JSON helper retries only once and then raises: `_json_completion` `retries=1` (`deep.py:315-347`). On 2026-08-14 20:53 the deep run gathered `learnings=19 sources=25 queries=4` but the final-report call returned prose ("We need answer final JSON with reportMarkdown string…"), so `error=LLM returned invalid JSON` and the report file was never written; the error propagated to `end errors=[...]` (`2026-08-14.md:1074, 1231`). With `deep_ok = bool(learnings) and not error` (`orchestrator.py:727`) the cycle reports a deep failure.

Search results do flow into learnings/sources when the stage runs: `_gather_hits` (`deep.py:557-564`) → `_distill` learnings → `result.sources` = visited URLs (`deep.py:610`). However the research-agent `search` tool returns only `kind<TAB>url<TAB>title`, no snippet text (`orchestrator.py:555`), so the agent-level loop cannot distill anything from it without an extra fetch.

Budget note: `max_queries=12` (`settings.yaml:109`) is never reached — every logged run stops at `queries=4` (breadth 3 + 3 follow-ups at depth 1), e.g. `2026-08-14.md:19`.

## Q3 — Browser extraction (browser-use vs browser-harness). Verdict: PARTIAL

It uses the browser-harness CLI, not the browser-use python package, and only in the separate browser-research step — never in the deep-research stage.

- `fetch_via_browser_harness` locates `shutil.which("browser-use") or shutil.which("browser-harness")` (`research/browser.py:118`), runs it as a subprocess (`browser.py:149-157`) with `BU_CDP_URL`/`BU_CDP_WS` (default `http://127.0.0.1:9224`, `browser.py:145-147`) and a script of `new_tab`/`wait_for_load`/`js(...)` calls (`browser.py:122-143`) — the browser-harness DSL, not the browser-use API. The binary exists at `~/.local/bin/browser-harness` and the submit-fallback traceback confirms it is the `browser_harness` pip package (`2026-08-14.md:407-412`).
- Browser research stage (settings `browser_research`, `settings.yaml:88-95`): `_browser_research` uses `BrowserResearcher.default(prefer_browser_harness=True)` (`orchestrator.py:628-649`, `browser.py:191-210`), i.e. browser-harness with `fetch_via_http` fallback. Works in production: `browser pages=['overview', 'discussion'] errors=none` every cycle (`2026-08-14.md:7`).
- Deep-research stage: `WebSource` defaults to `fetch_via_http` (`deep.py:272-273`; stdlib fetch `browser.py:171-188`) and the orchestrator constructs `WebSource()` with no browser fetch injected (`orchestrator.py:713-725`). Browser extraction is never used by the deep stage.
- Research-agent `fetch_url` tool only uses the browser when `browser_fetch` is injected — production passes none (only tests inject it, e.g. `tests/test_research_loop.py:57`), so it uses `fetch_via_http` (`orchestrator.py:531-538`).

## What works / what is broken

| Works | Broken |
|---|---|
| Kaggle kernel pull + real source cards with real refs (`memory/research-deep/source-wguesdon-*.md:2,9,13`) | Research agent loop: 286 invalid_json turns on 08-14; cycles burn 40 turns and never reach deep research |
| arXiv/GitHub/DDG searches return real hits; sources lists contain real URLs | DDG results are mostly off-topic; reports admit snippets were irrelevant and lean on snapshot + model knowledge |
| Learnings/sources/query counts logged and vary per run | Final report JSON frequently fails (`reportMarkdown`); on failure no report file is written (`2026-08-14.md:1074`) |
| Browser research step uses browser-harness CLI with CDP 9224, works (`errors=none`) | Deep stage and `fetch_url` never use the browser (http only) |
| Digest merge exists (`deep.py:614-617`) | Digest rarely survives: no `## Deep research digest` in `memory/research.md`; deep runs skipped entirely in 5 of 7 cycles on 08-14 |

## Recommended fixes (locations only)

- `src/kaggle_agent/agents/loop.py` `_next_action`: when two invalid-JSON turns happen, fall back to `must_first`-style deterministic tools or bail to the safety-net `_source_cards` instead of burning 40 turns.
- `src/kaggle_agent/orchestrator.py:588-590`: schedule `_deep_research` deterministically (run it in `_research` when `deep.enabled`, independent of the agent's tool choice).
- `src/kaggle_agent/research/deep.py:315-347` `_json_completion`: use a higher retry count for the 8192-token report call, or accept markdown output and wrap it (`_report_markdown` at `deep.py:577-591`).
- `src/kaggle_agent/orchestrator.py:713-725`: pass a browser-backed fetch into `WebSource()` so deep research uses browser-harness content instead of raw HTTP.
- `src/kaggle_agent/research/deep.py:455-476`: filter/re-rank `_fetch_all` content (e.g. drop hits whose text lacks the competition slug) before distillation, so learnings stop admitting irrelevance.
- `src/kaggle_agent/research/deep.py:569`: stamp report files with UTC to match daily logs (files are local-time: `03:37 UTC` run → `deep-20260814-090859.md`, UTC+5:30).
