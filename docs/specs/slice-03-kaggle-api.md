# Spec: Slice 3 — Kaggle API adapter

## Objective

Give the agent a **deep module** over the official Kaggle Public API so the daily cycle can:

1. Read **submission limits** (remaining allowance).
2. List **meta files** (CSV only; not full DICOM tree).
3. Download **small meta CSVs** (sample_submission, etc.).
4. Read **public leaderboard** top rows.
5. List **top public kernels** for the competition.
6. List **our recent submissions**.
7. Prepare **submit** behind `dry_run=True` by default (no accidental spend).

Submit stays **API-only** (never browser-harness). Browser is research HTML only (later slice).

## Tech stack

| Piece | Choice | Source |
|-------|--------|--------|
| Official client | `kaggle` PyPI package (CLI + `kaggle.api.KaggleApi`) | https://www.kaggle.com/docs/api |
| Verified version | 2.2.4 (probed on host via `uv tool run --from kaggle`) | package import |
| Auth | `~/.kaggle/kaggle.json` via `api.authenticate()` | same docs |
| HTTP | library-internal (we do not reimplement REST) | — |

## Module design (codebase-design)

**Module:** `kaggle_agent.kaggle_api`  
**Interface (small):**

| Method | Returns |
|--------|---------|
| `KaggleClient.connect()` | self, authenticated |
| `submission_limits(slug)` | `SubmissionLimits` |
| `list_meta_files(slug)` | root-level non-huge files (CSV/JSON/md) |
| `download_file(slug, name, dest)` | `Path` |
| `leaderboard(slug, top=n)` | `list[LeaderboardRow]` |
| `kernels(slug, top=n)` | `list[KernelRow]` |
| `submissions(slug, top=n)` | `list[SubmissionRow]` |
| `submit(slug, path, message, *, dry_run=True)` | `SubmitResult` |
| `research_snapshot(slug)` | fetch + plain dict for writers |

**Seam:** inject optional `api` object (official `KaggleApi` or test double).  
**Not in interface:** pagination tokens, SDK types, urllib.

**Orchestrator use:** `RESEARCH` phase calls `research_snapshot` → updates `memory/research.md` + budget fields in `memory/state.md`.

## Commands

```bash
cd ~/kaggle-agent
uv sync --extra dev
uv run pytest
uv run python scripts/run_daily.py --competition rsna_knee
# optional live smoke (needs ~/.kaggle/kaggle.json):
uv run pytest -m integration
```

## Testing strategy

| Level | What |
|-------|------|
| Unit | Fake API double; no network |
| Integration (optional marker) | Live `submission_limits` + `leaderboard` for pilot slug |

## Boundaries

- **Always:** default `dry_run=True` on submit; never download full DICOM tree; cite official API methods.
- **Ask first:** real submit (`dry_run=False`); Telegram still required in later slice.
- **Never:** browser-based submit; commit `kaggle.json`; store API secrets in memory md.

## Success criteria

1. Unit tests pass without network.
2. `research_snapshot` returns limits + LB + kernels for a fake API.
3. Dry-run cycle RESEARCH phase writes research.md when client available; degrades cleanly if auth missing.
4. Live probe (manual or integration) works with existing `~/.kaggle/kaggle.json`.

## Open questions

None for Slice 3 — pilot slug already in config.
