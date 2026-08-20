#!/usr/bin/env python3
"""Read Telegram getUpdates and write TELEGRAM_CHAT_ID into .env."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from kaggle_agent.config import load_dotenv  # noqa: E402


def load_env() -> None:
    """Compatibility wrapper around the canonical config loader."""
    load_dotenv(ROOT)


def main() -> int:
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("TELEGRAM_BOT_TOKEN not set in .env", file=sys.stderr)
        print(f"Expected file: {ENV}", file=sys.stderr)
        return 1
    # Sanity: getMe
    with urllib.request.urlopen(
        f"https://api.telegram.org/bot{token}/getMe", timeout=30
    ) as resp:
        me = json.loads(resp.read().decode())
    if not me.get("ok"):
        print("getMe failed — bad token?", me, file=sys.stderr)
        return 1
    uname = (me.get("result") or {}).get("username")
    print(f"bot ok: @{uname}")

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    if not data.get("ok"):
        print("getUpdates failed:", data, file=sys.stderr)
        return 1
    results = data.get("result") or []
    if not results:
        print("No messages yet (getUpdates empty).")
        print(f"1) Open https://t.me/{uname} and press Start, send: hi")
        print("2) Wait 2 seconds, re-run this script")
        print("Or just: bash scripts/start_telegram_bot.sh  (learns chat on first /help)")
        return 2

    chat_id = None
    for upd in reversed(results):
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat = msg.get("chat") or {}
        if chat.get("id") is not None:
            chat_id = str(chat["id"])
            print(f"found chat_id={chat_id} text={(msg.get('text') or '')[:50]!r}")
            break
    if not chat_id:
        print("updates present but no chat id", file=sys.stderr)
        return 1
    text = ENV.read_text(encoding="utf-8") if ENV.is_file() else ""
    if re.search(r"^TELEGRAM_CHAT_ID=.*$", text, re.M):
        text = re.sub(r"^TELEGRAM_CHAT_ID=.*$", f"TELEGRAM_CHAT_ID={chat_id}", text, flags=re.M)
    else:
        text = text.rstrip() + f"\nTELEGRAM_CHAT_ID={chat_id}\n"
    ENV.write_text(text, encoding="utf-8")
    ENV.chmod(0o600)
    print(f"wrote TELEGRAM_CHAT_ID to {ENV}")
    # confirm with a message
    body = json.dumps(
        {"chat_id": chat_id, "text": "kaggle-agent: chat id saved. Start the bot with scripts/start_telegram_bot.sh"}
    ).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        ok = json.loads(resp.read().decode()).get("ok")
    print("test message ok:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
