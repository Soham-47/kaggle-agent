# Kaggle submit (API first for notebooks)

## Order

Default (`submit.mcp: false` in `config/settings.yaml`):

1. Python Kaggle API
   - File contests: `competition_submit` of a CSV
   - Notebook contests: `kernels_push` + poll COMPLETE + `competition_submit_code`
2. browser-harness UI only if `browser_fallback` is on (needs signed-in Chrome)

MCP remains available if `submit.mcp: true`:

1. Kaggle MCP (`https://www.kaggle.com/mcp`)
   - Notebook comps: `create_code_competition_submission` (needs a COMPLETE kernel you own)
   - File comps: `start_competition_submission_upload` → blob PUT → `submit_to_competition`
2. API, then browser

Flags under `submit:`: `mcp`, `api`, `browser_fallback`. Live kernel push: `kernel.push: true` (still skipped on dry_run).

## Auth

Write tools need `~/.kaggle/access_token` (starts with `KGAT_`), not only `kaggle.json`.

```bash
source scripts/export_kaggle_mcp_token.sh
# sets KAGGLE_API_TOKEN for Grok MCP + agent
```

The agent also reads `~/.kaggle/access_token` if the env var is unset.

## Grok MCP

User config: `~/.grok/config.toml`  
Project: `kaggle-agent/.grok/config.toml`

```toml
[mcp_servers.kaggle]
url = "https://www.kaggle.com/mcp"
headers = { Authorization = "Bearer ${KAGGLE_API_TOKEN}" }
enabled = true
```
