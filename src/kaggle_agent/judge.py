"""Shared stage judges: deterministic first, optional LLM, verdict recorded.

One scaffold (`judge_stage`) used by the PLAN judge, the kernel judge, and
the flag-gated TRAIN/SUBMIT judge. Verdicts track consecutive identical
(ready, reason) pairs so stage loops can converge without relying on the
LLM's cooperation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.pipeline.validate import ValidationResult
from kaggle_agent.research.deep import _json_completion
from kaggle_agent.research.source_cards import step_is_junk, steps_implemented

_JUNK_STATUSES = frozenset({"error", "failed", "cancelled", "canceled"})

_PLAN_JUDGE_SYSTEM = (
    "You are the plan judge for one Kaggle experiment cycle. "
    "Return JSON {\"ready\": bool, \"reason\": str}. "
    "ready=false when the plan is a re-run of implemented steps, vague "
    "(\"improve the model\", \"tune hyperparameters\"), or copies a baseline "
    "already beaten by our public score. ready=true when the step is concrete "
    "and not yet implemented."
)

_TRAIN_JUDGE_SYSTEM = (
    "You are the train/submit judge for one Kaggle experiment cycle. "
    "Return JSON {\"ready\": bool, \"reason\": str}. "
    "ready=false when the kernel job failed, the output CSV is constant or "
    "pathological (all zeros, NaNs, identical rows), or the row count does not "
    "match the test set. ready=true when the output is plausible."
)


def new_judge_state() -> dict[str, Any]:
    return {"ready": None, "streak": 0, "last_reason": ""}


def record_verdict(judge_state: dict[str, Any], ready: bool, reason: str) -> None:
    """Track judge verdicts; the streak counts consecutive identical verdicts."""
    verdict = (bool(ready), reason)
    judge_state["streak"] = (
        judge_state["streak"] + 1 if verdict == judge_state.get("last_verdict") else 1
    )
    judge_state["last_verdict"] = verdict
    judge_state["ready"] = bool(ready)
    judge_state["last_reason"] = reason


def judge_stage(
    stage: str,
    *,
    state: dict[str, Any],
    deterministic: Callable[[], tuple[bool, str]],
    llm: Callable[[], tuple[bool, str]] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Run deterministic checks first; the LLM judge only when they pass.

    A failed deterministic check is a hard floor: no LLM override. Pass no
    `llm` callable when no model is available, and the deterministic verdict
    stands. Record every verdict in `state` and log one
    `judge <stage> ready=... reason=...` line.
    """
    ready, reason = deterministic()
    if ready and llm is not None:
        ready, reason = llm()
    record_verdict(state, ready, reason)
    if log is not None:
        log(f"judge {stage} ready={ready} reason={reason}")
    return ready, reason


def _steps_from_plan(plan_text: str) -> str:
    for line in (plan_text or "").splitlines():
        if line.lower().startswith("steps:"):
            return line.split(":", 1)[1].strip()
    return (plan_text or "").strip()


def _plan_deterministic(plan_text: str, methods: dict[str, Any]) -> tuple[bool, str]:
    steps = _steps_from_plan(plan_text)
    if not steps or step_is_junk(steps):
        return False, "no non-junk step"
    try:
        if steps_implemented(steps, methods):
            return False, "steps already implemented in methods.json"
    except Exception:  # noqa: BLE001
        pass
    return True, "deterministic"


def _plan_llm(
    zen: Any, model: str, plan_text: str, methods: dict[str, Any], our_score: str
) -> tuple[bool, str]:
    impl = "; ".join(methods.get("implement_steps") or []) or "none"
    user = (
        f"our_public_best={our_score}\n"
        f"already implemented: {impl}\n\n"
        f"plan:\n{plan_text}\n"
    )
    try:
        parsed = _json_completion(zen, model, _PLAN_JUDGE_SYSTEM, user, max_tokens=400)
    except Exception:  # noqa: BLE001
        return True, "judge-fail-open"
    ready = bool(parsed.get("ready"))
    return ready, str(parsed.get("reason") or ("ready" if ready else "not ready"))


def judge_plan(
    zen: Any,
    model: str,
    plan_text: str,
    methods: dict[str, Any],
    our_score: str,
    *,
    state: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """PLAN judge: novelty vs methods.json, deterministic first then LLM."""
    return judge_stage(
        "plan",
        state=state if state is not None else new_judge_state(),
        deterministic=lambda: _plan_deterministic(plan_text, methods),
        llm=(lambda: _plan_llm(zen, model, plan_text, methods, our_score))
        if zen is not None
        else None,
        log=log,
    )


def _kernel_deterministic(
    job_status: str,
    csv_check: ValidationResult,
    *,
    labels: list[str] | None = None,
    csv_path: Path | None = None,
) -> tuple[bool, str]:
    st = (job_status or "").lower().replace(" ", "")
    if st in _JUNK_STATUSES:
        return False, f"job status {st}"
    if not csv_check.ok:
        return False, "; ".join(csv_check.errors[:3])
    reason = f"job {st or 'none'} rows={csv_check.n_rows}"
    if labels is not None and csv_path is not None and csv_path.is_file():
        reason += f" stats={_csv_stats(csv_path, labels)}"
    return True, reason


def judge_kernel(
    job_status: str,
    csv_check: ValidationResult,
    *,
    state: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
    labels: list[str] | None = None,
    csv_path: Path | None = None,
) -> tuple[bool, str]:
    """CODE kernel judge: job log sanity + output CSV sanity (mechanical)."""
    return judge_stage(
        "kernel",
        state=state if state is not None else new_judge_state(),
        deterministic=lambda: _kernel_deterministic(
            job_status, csv_check, labels=labels, csv_path=csv_path
        ),
        log=log,
    )


def _csv_stats(path: Path, labels: list[str]) -> str:
    cols = {lab: [] for lab in labels}
    rows = 0
    invalid = 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows += 1
                for lab in labels:
                    try:
                        value = float(row.get(lab, ""))
                    except (TypeError, ValueError):
                        invalid += 1
                    else:
                        if math.isfinite(value):
                            cols[lab].append(value)
                        else:
                            invalid += 1
    except Exception:  # noqa: BLE001
        return "rows=0"
    if rows == 0:
        return "rows=0"
    finite = list(cols.values())
    means = [sum(f) / len(f) if f else 0.0 for f in finite]
    constant = any(
        not f or all(x == f[0] for x in f)
        for v, f in zip(cols.values(), finite)
    )
    mean = sum(means) / len(means) if means else 0.0
    low = min(means) if means else 0.0
    high = max(means) if means else 0.0
    return (
        f"rows={rows} nan={invalid} mean={mean:.3f} "
        f"min={low:.3f} max={high:.3f} constant={constant}"
    )


def judge_train_llm(
    zen: Any,
    model: str,
    job_status: str,
    csv_path: Path,
    labels: list[str],
    our_score: str,
    *,
    state: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """TRAIN/SUBMIT judge behind a flag: LLM layer over the mechanical verdict."""
    stats = _csv_stats(csv_path, labels)
    user = (
        f"our_public_best={our_score}\n"
        f"kernel job status={job_status}\n"
        f"output CSV stats: {stats}\n"
    )

    def llm() -> tuple[bool, str]:
        try:
            parsed = _json_completion(zen, model, _TRAIN_JUDGE_SYSTEM, user, max_tokens=400)
        except Exception:  # noqa: BLE001
            return True, "judge-fail-open"
        ready = bool(parsed.get("ready"))
        return ready, str(parsed.get("reason") or ("ready" if ready else "not ready"))

    return judge_stage(
        "train",
        state=state if state is not None else new_judge_state(),
        deterministic=lambda: (True, "mechanical pass"),
        llm=llm,
        log=log,
    )
