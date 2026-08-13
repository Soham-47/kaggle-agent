# Review: Slice 7 — Telegram approve + submit gate

### Context
- [x] Spec: `docs/specs/slice-07-telegram-submit.md`
- [x] Simplification pass applied after implementation

### Correctness
- [x] Dry-run submit never calls real `competition_submit` with dry_run=False
- [x] Live submit blocked without `pending.status == approved`
- [x] `/approve` / `/reject` / `/pause` / `/resume` / `/budget` update markdown state
- [x] Unauthorized chat IDs rejected when allowed_chat_id set
- [x] Tests: **41 passed** offline

### Readability
- [x] Command handlers extracted; submit dry vs real split
- [x] Pending submit is one markdown file (`pending_submit.md`)

### Architecture
- [x] Deep modules: `notify.telegram`, `notify.commands`, `submit.pending`
- [x] Injectable Telegram + Kaggle clients for tests
- [x] Orchestrator owns phase wiring only

### Security
- [x] Tokens only via env (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`)
- [x] Chat ID filter on bot process
- [x] No secrets in memory md
- [x] require_telegram_approve default true

### Performance
- [x] Bot long-poll is separate process (not in daily cycle)
- [x] Notify failures fail-soft

### Findings
| Severity | Item | Status |
|----------|------|--------|
| **Required** | Dead no-op exp check in `_submit` | Fixed in simplification |
| **Required** | Silly `_submit_dry` return hack | Fixed |
| **Optional** | Live cycle still needs human `/approve` between cycles if TELEGRAM_APPROVE resets pending to pending each run | By design: approve then re-run cycle, or later wait-loop |
| **Optional** | HEAL is placeholder until slice 8 | OK |
| **FYI** | Full Telegram Bot API not used (no keyboards) | OK for v1 |

### Verdict
- [x] **Approve** — ready for Slice 8 (heal + cron) or production bot wiring
