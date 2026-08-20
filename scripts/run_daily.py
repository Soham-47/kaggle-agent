#!/usr/bin/env python3
"""Cron entrypoint: run one orchestrator cycle."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install: add src to path
ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from kaggle_agent.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
