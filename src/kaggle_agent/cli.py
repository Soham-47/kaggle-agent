"""CLI entry for kaggle-agent."""

from __future__ import annotations

import argparse
import json
import os
import sys

from kaggle_agent.config import load_dotenv
from kaggle_agent.orchestrator import run_daily
from kaggle_agent.paths import repo_root


def main(argv: list[str] | None = None) -> int:
    load_dotenv(repo_root())
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "dashboard":
        return _dashboard(argv[1:])
    if argv and argv[0] == "evals":
        return _evals(argv[1:])
    if argv and argv[0] == "onboard":
        return _onboard(argv[1:])
    if argv and argv[0] in {"run", "/run"}:
        return _run(argv[1:])

    p = argparse.ArgumentParser(description="kaggle-agent daily cycle")
    p.add_argument("--competition", default=None)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--assume-approved",
        action="store_true",
        help="Treat this live cycle as already approved (Telegram /run).",
    )
    p.add_argument(
        "--skip",
        default=None,
        help="Comma-separated phases to skip, e.g. research,plan,code. "
        "Also read from KAGGLE_AGENT_SKIP_PHASES.",
    )
    args = p.parse_args(argv)
    skip = args.skip or os.environ.get("KAGGLE_AGENT_SKIP_PHASES", "")
    skip_phases = frozenset(
        s.strip().upper() for s in skip.split(",") if s.strip()
    )

    r = run_daily(
        args.competition,
        dry_run=args.dry_run,
        assume_approved=args.assume_approved,
        skip_phases=skip_phases or None,
    )
    if r.skipped:
        print(f"skipped: {r.skip_reason}")
        return 0
    print(
        f"ok competition={r.competition} dry_run={r.dry_run} "
        f"phases={len(r.phases_run)} ctx={r.context_sections} exp={r.experiment_id} "
        f"kaggle={r.kaggle_ok} browser={r.browser_ok} code={r.code_ok} smoke={r.smoke_ok} "
        f"kernel={r.kernel_ok} validate={r.validate_ok} approve={r.approve_ok} "
        f"submit={r.submit_ok} heal={r.heal_decision}"
        f" research_verified={r.research_verified} plan_verified={r.plan_verified}"
        f" code_verified={r.code_verified}"
    )
    if r.smoke_path:
        print(f"smoke_csv={r.smoke_path}")
    if r.kernel_path:
        print(f"kernel_pkg={r.kernel_path} ref={r.kernel_ref}")
    if r.candidate_csv:
        print(f"candidate_csv={r.candidate_csv}")

    if r.plan_text:
        print("--- plan ---")
        print(r.plan_text[:2000])
    if r.waiting_approve:
        print(
            "waiting_approve: send /yes then /run to submit",
            file=sys.stderr,
        )
        return 0  # not a hard failure
    hard = r.hard_errors
    if hard:
        print("errors:", "; ".join(hard), file=sys.stderr)
        return 1
    return 0


def _dashboard(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="kaggle-agent dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7777)
    args = p.parse_args(argv)
    from kaggle_agent.ops.dashboard import serve

    serve(repo_root(), host=args.host, port=args.port)
    return 0


def _run(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="kaggle-agent run")
    p.add_argument("mode", nargs="?", default="live", choices=["live", "dry"])
    args = p.parse_args(argv)
    from kaggle_agent.notify.run_agent import start_agent_cycle

    res = start_agent_cycle(dry_run=args.mode == "dry", background=True)
    print(res.message)
    return 0 if res.ok else 1


def _evals(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="kaggle-agent evals")
    p.parse_args(argv)
    from kaggle_agent.ops.evals import evaluate_cycle, persist_report

    report = evaluate_cycle(repo_root())
    persist_report(repo_root(), report)
    print(json.dumps(report, indent=2))
    if report["passed"]:
        print("GATE OPEN")
        return 0
    print("GATE CLOSED", file=sys.stderr)
    return 1


def _onboard_slug(root, slug):
    from kaggle_agent.autonomy.onboard import CompetitionBootstrapper
    from kaggle_agent.kaggle_api import KaggleClient

    return CompetitionBootstrapper(root, KaggleClient().connect()).onboard(slug)


def _onboard(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="kaggle-agent onboard")
    p.add_argument("competition_slug")
    args = p.parse_args(argv)
    result = _onboard_slug(repo_root(), args.competition_slug)
    if result.contract is None:
        print(
            f"onboard={result.outcome.state.value}: {result.outcome.summary}",
            file=sys.stderr,
        )
        return 2 if result.outcome.state.value == "needs_authority" else 1
    print(
        f"onboard=success competition={result.contract.raw['id']} "
        f"contract_hash={result.contract.compatibility_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
