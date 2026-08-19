# Inference Engineering in 100 Days

A GPU-first, hands-on course on LLM inference engineering. Ten phases,
one lesson per day. Lessons run on CPU in smoke mode and use a CUDA GPU
when one is available. All lesson prose is original; the topic map and
sources live in `ROADMAP.md` and `sources/sources.md`.

## Structure

| Path | What it is |
|------|------------|
| `days/day-001/` … `days/day-100/` | One self-contained lesson folder per day |
| `shared/` | Small simulation library that lessons import |
| `scripts/check_day.py` | Validates a day folder and runs its lesson |
| `scripts/progress.py` | Prints roadmap progress per phase |
| `ROADMAP.md` | The 100-day topic map |
| `sources/sources.md` | The three references and how they map to days |
| `tests/` | Suite for the shared library, scripts, and roadmap |

## Setup

From the repository root:

```
uv sync --project inference-course
```

This creates the course virtualenv at `inference-course/.venv` with
`numpy`, `torch`, and `pytest`.

## Commands

Run the course test suite:

```
uv run --directory inference-course pytest
```

Validate a day folder and run its lesson in smoke mode:

```
uv run --directory inference-course python scripts/check_day.py --day 1 --smoke
```

`check_day.py` also accepts `--day N` without `--smoke`, which runs the
lesson in full mode.

## Day-folder contract

Each `days/day-XXX/` folder contains exactly four files:

1. `README.md` with these sections in order: `# Day N — Topic`,
   `## Goal`, `## Prerequisites`, `## Concept`, `## Experiment`,
   `## Expected observations`, `## Metric`, `## Sources`,
   `## Hardware notes`, `## Reflection prompts`. The Metric section
   states one concrete quantity the experiment measures; the Sources
   section lists the reference links used.
2. `lesson.py` — runnable and self-contained. It accepts `--smoke`
   (small sizes, CPU, no downloads) and runs full mode on GPU when
   available, and prints the observed metric values.
3. `exercise.md` — one exercise with acceptance hints.
4. `reflection.md` — 3-5 reflection questions with blank answer lines.

Lessons import only from `shared/` (`env`, `metrics`, `kv_cache`,
`batching`, `quantize`, `speculative`).

## GPU notes

- The test suite runs on CPU; CUDA-only code paths are guarded by
  `shared/env.py` and marked `@pytest.mark.gpu`.
- Smoke mode never downloads models and never requires a GPU.
- Use `shared/env.py` (`has_cuda`, `device`, `device_name`,
  `vram_available_gb`) to detect the GPU at runtime.