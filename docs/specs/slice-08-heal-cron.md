# Spec: Slice 8, heal policy + cron + kernel resume

## Objective

1. Heal: after each cycle, choose next action: `tune` → `recipe` → `new` → `pause` based on score progress and attempt counts.
2. Kernel resume: if a pushed kernel is still RUNNING, next cron polls it instead of starting a second push.
3. Cron: document + install helper for daily `run_daily.py` with flock via existing `RunLock`.

## Success criteria

- Unit tests cover heal ladder and kernel-job resume path (fakes).
- `memory/heal.md` records streak / decision.
- `memory/kernel_job.md` tracks in-flight kernel ref + status.
- `docs/cron.md` + `scripts/install_cron.sh` install a daily job.
- Pause after `max_no_improve_days` sets `state.paused` and notifies Telegram if available.

## Boundaries

- Never force-push a second kernel while one is RUNNING for the same competition.
- Cron must not double-run (RunLock already).
