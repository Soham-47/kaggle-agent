# Spec: Slice 4-5, browser research + pipeline smoke

## Objective

1. Slice 4: scrape competition overview/discussion text (when the API is thin) via an injectable page fetcher. The default production path uses browser-harness. Never submit via browser.
2. Slice 5: ship a schema-correct baseline pipeline and local smoke that writes a valid submission CSV (constant probs) without DICOM train.

## Success criteria

- Unit tests offline with fake page HTML + fake pipeline I/O.
- `RESEARCH` appends a `## Browser` section into `memory/research.md` when scrape runs.
- `LOCAL_SMOKE` produces `competitions/rsna_knee/submissions/*.csv` with exact 12-label header including `Baker's`.
- Smoke validates row count, id column, probability range [0,1].
- Dry-run cycle completes with `smoke_ok` / `browser` status fields.

## Boundaries

- Always: API for submit; browser for read-only research pages.
- Never: download full DICOM locally in smoke; auto real submit.
