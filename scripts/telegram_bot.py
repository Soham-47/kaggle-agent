#!/usr/bin/env python3
"""Long-poll Telegram bot for approve/pause/status/run commands."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

_env = ROOT / ".env"
if _env.is_file():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))

from kaggle_agent.notify.commands import process_updates  # noqa: E402
from kaggle_agent.notify.run_agent import start_agent_cycle_async  # noqa: E402
from kaggle_agent.notify.telegram import TelegramClient, TelegramError  # noqa: E402


def _persist_chat_id(chat_id: str) -> None:
    os.environ["TELEGRAM_CHAT_ID"] = chat_id
    path = ROOT / ".env"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if re.search(r"^TELEGRAM_CHAT_ID=.*$", text, re.M):
        text = re.sub(
            r"^TELEGRAM_CHAT_ID=.*$", f"TELEGRAM_CHAT_ID={chat_id}", text, flags=re.M
        )
    else:
        text = text.rstrip() + f"\nTELEGRAM_CHAT_ID={chat_id}\n"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def main() -> int:
    client = TelegramClient.from_env()
    if client is None:
        print("Set TELEGRAM_BOT_TOKEN (e.g. in .env)", file=sys.stderr)
        return 1

    offset: int | None = None
    print("telegram bot polling… Ctrl+C to stop")
    print(f"token ok; chat_id={client.default_chat_id or '(will learn on first /command)'}")

    if client.default_chat_id:
        try:
            client.send_message(
                "Kaggle agent bot is online.\n\n"
                "Try /help for commands, /status for state, /run for a safe dry cycle."
            )
        except TelegramError as exc:
            print(f"startup notify failed: {exc}", file=sys.stderr)

    while True:
        try:
            updates = client.get_updates(offset=offset, timeout=25)
        except (TelegramError, OSError, TimeoutError) as exc:
            print(f"poll error: {exc}", file=sys.stderr)
            time.sleep(5)
            continue
        except Exception as exc:  # noqa: BLE001 — poller must never die
            print(f"poll error (unexpected): {exc}", file=sys.stderr)
            time.sleep(5)
            continue

        for upd in updates:
            uid = upd.get("update_id")
            if isinstance(uid, int):
                offset = uid + 1

        allowed = client.default_chat_id or None
        results = process_updates(
            updates,
            root=ROOT,
            allowed_chat_id=allowed,
            lock_first_chat=True,
        )
        for r in results:
            chat = r.chat_id or client.default_chat_id
            if chat and not client.default_chat_id:
                client.set_default_chat(chat)
                _persist_chat_id(chat)
                print(f"learned chat_id={chat}")
            if not chat:
                print("reply skipped: no chat_id", file=sys.stderr)
                continue
            try:
                client.send_message(r.reply, chat_id=chat)
            except TelegramError as exc:
                print(f"send error: {exc}", file=sys.stderr)
                continue

            if r.start_cycle:
                dry = r.cycle_dry_run

                def _done(res, chat_id=chat, dry_run=dry):
                    try:
                        tag = "dry" if dry_run else "live"
                        head = "Cycle finished" if res.ok else "Cycle finished with problems"
                        client.send_message(
                            f"{head} ({tag}).\n\n{res.message}",
                            chat_id=chat_id,
                        )
                    except TelegramError as exc:
                        print(f"done notify failed: {exc}", file=sys.stderr)

                start = start_agent_cycle_async(
                    root=ROOT, dry_run=dry, on_done=_done
                )
                try:
                    client.send_message(start.message, chat_id=chat)
                except TelegramError as exc:
                    print(f"run notify failed: {exc}", file=sys.stderr)

        if not updates:
            time.sleep(0.2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nbye")
        raise SystemExit(0)
