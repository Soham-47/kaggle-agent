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

## Restricted AUTO_SAFE profile

`config/profiles/controlled-auto-safe.yaml` is an explicit operator override.
It is not loaded by default and does not change the checked-in safe defaults.
Use it only with a reviewed disposable state root:

```bash
kaggle-agent supervisor --profile controlled-auto-safe --competition <id>
```

The profile limits repairs to one per cycle and two per day, allows at most
two implementation attempts per incident, keeps dependency changes disabled,
and retains strict protected-path, specification-review, code-review, and
full-test gates. Unresolved external actions still block promotion.
