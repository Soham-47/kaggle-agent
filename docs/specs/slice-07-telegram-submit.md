# Spec: Slice 7, Telegram approve + Kaggle submit gate

## Objective

1. Notify via Telegram Bot API (stdlib HTTP).
2. Two-way commands: `/status`, `/approve`, `/reject`, `/pause`, `/resume`, `/budget`, `/help`.
3. Real Kaggle submit only after `/approve <exp_id>` when `require_telegram_approve: true`.
4. Dry-run never spends a submission (`KaggleClient.submit(..., dry_run=True)`).

## Success criteria

- Unit tests with FakeTelegram + FakeKaggleApi (no network).
- TELEGRAM_APPROVE writes `memory/pending_submit.md` + optional notify.
- SUBMIT blocked without approval when required.
- `/approve` then submit path works in unit test with injects.
- secrets only from `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` env vars.

## Boundaries

- Always: API submit only; never browser submit.
- Never: commit tokens; auto real submit without approve when require_approve true.
