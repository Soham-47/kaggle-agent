"""CLI entry for kaggle-agent."""

from __future__ import annotations

import argparse
import sys

from kaggle_agent.orchestrator import run_daily


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="kaggle-agent daily cycle")
    p.add_argument("--competition", default=None)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    args = p.parse_args(argv)

    r = run_daily(args.competition, dry_run=args.dry_run)
    if r.skipped:
        print(f"skipped: {r.skip_reason}")
        return 0
    print(
        f"ok competition={r.competition} dry_run={r.dry_run} "
        f"phases={len(r.phases_run)} ctx={r.context_sections} exp={r.experiment_id} "
        f"kaggle={r.kaggle_ok} browser={r.browser_ok} code={r.code_ok} smoke={r.smoke_ok} "
        f"kernel={r.kernel_ok} validate={r.validate_ok} approve={r.approve_ok} "
        f"submit={r.submit_ok} heal={r.heal_decision}"
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
            "waiting_approve: send /yes then /run live to submit",
            file=sys.stderr,
        )
        return 0  # not a hard failure
    hard = r.hard_errors
    if hard:
        print("errors:", "; ".join(hard), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
