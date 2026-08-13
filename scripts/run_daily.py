#!/usr/bin/env python3
"""Cron entrypoint: run one orchestrator cycle."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running without install: add src to path
ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

# Load .env for cron / bare shells
_env = ROOT / ".env"
if _env.is_file():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))

from kaggle_agent.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
