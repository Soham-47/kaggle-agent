# MEMORY

Durable facts only. Keep short. Contest-specific numbers live under Active contest.

## User

- Decision maker: Soham. Implement only approved slices unless autonomy is on for a run.
- LLM: OpenCode Zen (`OPENCODE_API_KEY`). Models: text + multimodal when the contest needs images.
- Train: Kaggle Kernels. Local = smoke only.
- Submit: Kaggle API. For notebooks-only contests use `kernels_push` + `competition_submit_code`. Keep `submit.mcp: false` unless you own a COMPLETE kernel the MCP can attach. Never browser-submit.
- Real submit requires `memory/pending_submit.md` status=approved (`/yes` or `/approve <exp_id>`). Dry-run never spends quota.
- Telegram bot: `scripts/telegram_bot.py` + env `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`.
- Research: Kaggle API first (limits, LB, kernels, meta CSV). Then browser-harness for overview/discussion HTML. Then one method card per top public kernel. Then DeepResearcher. Fail-soft. Never use the browser to submit.
- Kernel: always build a notebook package under `competitions/<id>/notebooks/<exp>/`. Push only if `kernel.push: true` and the cycle is not dry.
- Kernel resume: if `memory/kernel_job.md` is still RUNNING, the next cycle polls that job (no second push).
- Heal: tune → recipe → new → pause (`memory/heal.md`). Pause sets `state.paused`.
- Cron: `bash scripts/install_cron.sh` — see `docs/cron.md`.

## Goals

1. Autonomous daily loop with a small surface area.
2. Valid schema baseline, then beat personal best without burning quota.
3. Self-heal: tune → recipe → new → pause.

## Active contest

Swap this block when you change `default_competition`.

- id: rsna_knee
- slug: rsna-knee-abnormality-detection
- metric: macro AUC (rank-based). A constant probability file scores about 0.500.
- public_score: 0.526
- experiment: 20260813-cards-fix / submission 55486807 (same 0.526 as 55485145)
- submit header: keep `Baker's` apostrophe
- host rejects P100; `kernel.enable_gpu` stays false unless the host allows another accelerator

## Lessons (any contest)

- Kaggle often mounts competition data at `/kaggle/input/competitions/<slug>/` (note the `competitions/` prefix). Some kernels see DICOMs only and no CSVs. Embed test IDs at build time and also discover IDs from study folders on the hidden rerun.
- `write_kernel_package` must raise if no study IDs exist. A file built from fake IDs is a scoring error.
- Do not `import pandas` in local agent paths. It is not in local deps. Kernels have it. Use stdlib `csv` locally.
- Sibling `.py` files are not importable inside Kaggle notebooks. Inline the recipe.
- Pending-submit reuse must not carry a stale `kernel_path` / `kernel_ref`. Reset those to `none` before a new live package.
- Browser research uses a headed profile from `scripts/start_research_chrome.sh` (`~/.local/share/kaggle-agent/chrome`, `BU_CDP_URL=http://127.0.0.1:9224`). Sign in to Kaggle once in that window. Do not attach to daily Chrome 9222 (often stale after an upgrade). Headless `/tmp/kaggle-agent-chrome` has no Google session.
- Stale `memory/run.lock` after a killed run blocks the next cycle. Remove the file and set `lock_held: false`.
- CODE should implement method cards: attach public weight datasets, rank-average members, do not probability-mean AUC ensembles.
