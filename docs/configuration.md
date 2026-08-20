# Configuration

`config/settings.yaml` contains shared defaults and supervisor policy. It does
not contain secrets. The public configuration intentionally leaves
`default_competition` unset; pass `--competition` or set it after initializing
a contest.

Each competition has a YAML contract under `config/competitions/`. Start from
`_template.yaml`. The contract owns the metric, labels, submission schema,
workspace path, training backend, and submission mode.

Supported environment overrides include:

- `KAGGLE_AGENT_STATE_ROOT` for mutable worker state.
- `KAGGLE_AGENT_SUPERVISOR_DIR` for supervisor state and managed generations.
- `DEEPSEEK_API_KEY` for official DeepSeek calls.
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` for optional controls.
