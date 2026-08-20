"""Parse and handle Telegram bot commands against agent memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.loop import load_loop
from kaggle_agent.config import load_settings
from kaggle_agent.paths import repo_root
from kaggle_agent.state_md import load_state, save_state
from kaggle_agent.submit.pending import load_pending, set_decision
from kaggle_agent.supervisor.commands import SupervisorCommandQueue
from kaggle_agent.supervisor.state import RuntimeLayout

HELP = """Kaggle agent — what you can do

Run
• /run — full live loop (research until cards, N train slices, one submit)
• /run dry — practice loop, no real Kaggle submit

/run counts as submit approval. I will not wait for /yes.

Check
• /status — phase, budget, pending submit, loop next_n / last_n / last_score

Approve
• /yes — optional; /run already counts as approval
• /no — reject a pending candidate
• /approve <exp_id> — approve a specific run
• /reject <exp_id> — reject a specific run

Control
• /pause — stop new cycles
• /resume — allow cycles again
• /budget <n> — max auto proposals per day (e.g. /budget 2)

Help
• /help — this message

Typical flow: send /run. That loops then submits once."""


@dataclass
class CommandResult:
    ok: bool
    reply: str
    chat_id: str = ""
    start_cycle: bool = False
    cycle_dry_run: bool = True


def handle_command(text: str, *, root: Path | None = None) -> CommandResult:
    text = (text or "").strip()
    if not text.startswith("/"):
        return CommandResult(
            ok=False,
            reply="Send a command (starts with /).\n\nType /help for the full list.",
        )

    parts = text.split()
    cmd = parts[0].lower().split("@")[0]
    args = parts[1:]

    if cmd in {"/help", "/start"}:
        return CommandResult(ok=True, reply=HELP)
    if cmd == "/status":
        return CommandResult(ok=True, reply=_status_text(root))
    if cmd == "/run":
        return _run_cmd(args, root)
    if cmd == "/pause":
        return _set_paused(
            True,
            root,
            "Agent paused.\n\nNew cycles will be skipped until you send /resume.",
        )
    if cmd == "/resume":
        return _set_paused(
            False,
            root,
            "Agent resumed.\n\nYou can /run again (or wait for the daily cron).",
        )
    if cmd == "/budget":
        if not args or not args[0].isdigit():
            return CommandResult(
                ok=False,
                reply="Set how many submit proposals per day.\n\nUsage: /budget 2",
            )
        state = load_state(root)
        state.max_proposals = str(max(0, int(args[0])))
        save_state(state, root)
        return CommandResult(
            ok=True,
            reply=(
                f"Daily proposal cap set to {state.max_proposals}.\n\n"
                f"Used today: {state.proposals_used} (date={state.budget_date})."
            ),
        )
    if cmd in {"/approve", "/yes", "/ok"}:
        return _decide(args[0] if args else "latest", True, root)
    if cmd in {"/reject", "/no"}:
        return _decide(args[0] if args else "latest", False, root)
    return CommandResult(
        ok=False,
        reply=f"Unknown command: {cmd}\n\nType /help for the list.",
    )


def _run_cmd(args: list[str], root: Path | None) -> CommandResult:
    dry = False
    if args:
        flag = args[0].lower()
        if flag in {"live", "prod", "real", "--live", "--no-dry-run"}:
            dry = False
        elif flag in {"dry", "dry-run", "--dry-run"}:
            dry = True
        else:
            return CommandResult(
                ok=False,
                reply=(
                    "Usage:\n"
                    "• /run — full live loop (research until cards, N slices, one submit)\n"
                    "• /run dry — practice loop (no real submit)"
                ),
            )
    state = load_state(root)
    supervisor = _supervisor_queue(root)
    if supervisor is not None and supervisor.paused():
        return CommandResult(ok=False, reply="Supervisor is paused. Send /resume, then /run again.")
    if state.paused:
        return CommandResult(
            ok=False,
            reply="Agent is paused.\n\nSend /resume, then /run again.",
        )
    if dry:
        reply = (
            "Starting a practice run (/run dry)…\n\n"
            "Research until cards, then N train slices. "
            "It will not spend a Kaggle submission.\n\n"
            "I'll message you when it finishes. /status anytime."
        )
    else:
        reply = (
            "Starting a full live loop…\n\n"
            "Research until cards → N train slices → one submit.\n"
            "/run counts as your approval. I will not wait for /yes.\n\n"
            "I'll message you when it finishes."
        )
    if supervisor is not None:
        supervisor.enqueue("run", {"dry_run": dry})
        return CommandResult(
            ok=True,
            reply=(
                f"Supervisor queued a {'practice' if dry else 'live'} run.\n\n"
                "The supervisor is the only process that can launch the worker.\n"
                "Use /status for ownership and progress."
            ),
            cycle_dry_run=dry,
        )
    return CommandResult(
        ok=True,
        reply=reply,
        start_cycle=True,
        cycle_dry_run=dry,
    )


def _set_paused(paused: bool, root: Path | None, reply: str) -> CommandResult:
    supervisor = _supervisor_queue(root)
    if supervisor is not None:
        supervisor.enqueue("pause" if paused else "resume")
        return CommandResult(ok=True, reply=f"Supervisor queued control change. {reply}")
    state = load_state(root)
    state.paused = paused
    save_state(state, root)
    return CommandResult(ok=True, reply=reply)


def _decide(exp: str, approved: bool, root: Path | None) -> CommandResult:
    try:
        pending = set_decision(exp, approved=approved, root=root)
    except ValueError as exc:
        return CommandResult(
            ok=False,
            reply=(
                f"Could not update approval.\n\n{exc}\n\n"
                "Check /status for the current pending exp id, "
                "or use /yes for the latest."
            ),
        )
    state = load_state(root)
    if approved:
        state.pending_approve = pending.exp_id
        state.note = f"approved:{pending.exp_id}"
        # Unpause if heal had paused while waiting
        if state.paused and "pause" in (state.note or "").lower():
            pass
        save_state(state, root)
        reply = (
            "Submit approved.\n\n"
            f"Experiment: {pending.exp_id}\n"
            f"CSV: {pending.csv_path}\n"
            f"Competition: {pending.competition}\n\n"
            "What this means:\n"
            "The file is cleared for a real Kaggle submit.\n\n"
            "Next step:\n"
            "Send /run\n\n"
            "That loop will reuse this approved candidate and call "
            "the Kaggle submit API."
        )
        return CommandResult(ok=True, reply=reply)

    state.pending_approve = "none"
    state.note = f"rejected:{pending.exp_id}"
    save_state(state, root)
    return CommandResult(
        ok=True,
        reply=(
            "Submit rejected.\n\n"
            f"Experiment: {pending.exp_id}\n"
            f"CSV: {pending.csv_path}\n\n"
            "Nothing will be sent to Kaggle for this candidate.\n"
            "You can /run again to build a new one."
        ),
    )


def _status_text(root: Path | None) -> str:
    state = load_state(root)
    pending = load_pending(root)
    loop = load_loop(root)
    supervisor = _supervisor_queue(root)
    pause = "yes — send /resume" if state.paused else "no"
    return "\n".join(
        [
            "Agent status",
            "",
            f"Phase: {state.phase}",
            f"Paused: {pause}",
            f"Competition: {state.competition}",
            f"Active experiment: {state.active_experiment}",
            f"Budget today: {state.proposals_used} used / {state.max_proposals} max "
            f"(date {state.budget_date})",
            f"Personal best score: {state.public_best}",
            "",
            "Loop",
            f"  next_n: {loop.next_n}",
            f"  last_n: {loop.last_n}",
            f"  last_score: {loop.last_score}",
            "",
            "Pending submit",
            f"  Status: {pending.status}",
            f"  Exp: {pending.exp_id}",
            f"  CSV: {pending.csv_path}",
            f"  Competition: {pending.competition}",
            "",
            f"Note: {state.note}",
            "",
            "Supervisor",
            f"  Enabled: {'yes' if supervisor is not None else 'no'}",
            f"  Queued commands: {len(supervisor.pending()) if supervisor is not None else 0}",
            f"  Control paused: {'yes' if supervisor is not None and supervisor.paused() else 'no'}",
            "",
            "Tips: /run · /run dry · /yes · /no · /help",
        ]
    )


def _supervisor_queue(root: Path | None) -> SupervisorCommandQueue | None:
    selected = (root or repo_root()).resolve()
    try:
        if not load_settings(selected).supervisor_config().enabled:
            return None
    except Exception:  # noqa: BLE001
        return None
    return SupervisorCommandQueue(RuntimeLayout.for_repo(selected))


def process_updates(
    updates: list[dict],
    *,
    root: Path | None = None,
    allowed_chat_id: str | None = None,
    lock_first_chat: bool = True,
) -> list[CommandResult]:
    """Handle commands. If allowed_chat_id is empty and lock_first_chat, first chat wins."""
    results: list[CommandResult] = []
    locked = str(allowed_chat_id) if allowed_chat_id else ""
    for upd in updates:
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat_id = str((msg.get("chat") or {}).get("id", ""))
        text = msg.get("text") or ""
        if not chat_id or not text.startswith("/"):
            continue
        if locked and chat_id != locked:
            results.append(
                CommandResult(
                    ok=False,
                    reply=(
                        "This bot only accepts commands from the linked chat.\n\n"
                        "If this is your account, check TELEGRAM_CHAT_ID in .env."
                    ),
                    chat_id=chat_id,
                )
            )
            continue
        if not locked and lock_first_chat:
            locked = chat_id
        result = handle_command(text, root=root)
        result.chat_id = chat_id
        results.append(result)
    return results
