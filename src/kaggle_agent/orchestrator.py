"""Daily cycle: Kaggle research, browser research, pipeline smoke, optional Zen PLAN."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_agent.code.workspace import ensure_pipeline_ready
from kaggle_agent.config import CompetitionConfig, Settings, load_competition, load_settings
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.llm.router import ModelRouter
from kaggle_agent.memory.ingest import build_context_pack
from kaggle_agent.memory.write import append_daily_log, write_experiment
from kaggle_agent.notify.telegram import SupportsTelegram, TelegramClient
from kaggle_agent.paths import memory_dir
from kaggle_agent.pipeline.validate import validate_submission_csv
from kaggle_agent.research.apply_snapshot import apply_kaggle_research
from kaggle_agent.research.browser import (
    BrowserResearcher,
    FetchFn,
    merge_browser_into_research_md,
)
from kaggle_agent.research.source_cards import run_source_card_research, write_methods_sidecar
from kaggle_agent.research.deep import (
    ArxivSource,
    DeepResearcher,
    GithubSource,
    KaggleSource,
    WebSource,
)
from kaggle_agent.state_md import AgentState, RunLock, load_state, save_state
from kaggle_agent.kaggle_api.mcp_submit import submit_via_mcp
from kaggle_agent.kaggle_api.submit_ops import normalize_kernel_ref
from kaggle_agent.submit.browser_submit import (
    BrowserSubmitFn,
    BrowserSubmitRequest,
    submit_via_browser,
)
from kaggle_agent.submit.pending import (
    load_pending,
    mark_submitted,
    request_approval,
    save_pending,
    usable_approval,
)
from kaggle_agent.heal.policy import decide_next, load_heal, save_heal
from kaggle_agent.train.kernel_job import load_kernel_job
from kaggle_agent.train.kernel_runner import run_kernel_phase
from kaggle_agent.train.local_smoke import run_competition_smoke
from kaggle_agent.train.notebook_builder import write_kernel_package

PLAN_SYSTEM = """You plan the next Kaggle experiment. Be brief.
Reply with exactly 3 lines:
hypothesis: <one sentence>
approach: baseline|tune|recipe|new
steps: <semicolon-separated minimal steps>
Prefer the copyable next step from method cards (research-deep/source-*.md
and the Deep research digest) over a constant-score baseline. No code."""

CODE_SYSTEM = """You are the coding agent for this Kaggle cycle.
Read the method cards. Reply with at most 8 short lines:
1. Which public datasets/models to attach
2. How to find hidden test IDs
3. How to combine scores (rank-mean vs other)
4. What not to copy
Do not invent a new architecture. Do not include secrets."""

DEFAULT_HYPOTHESIS = "dry-run default: schema-valid 0.5 baseline then improve"

TRAIN_SLICE_PHASES = ("PLAN", "CODE", "LOCAL_SMOKE", "KERNEL_TRAIN", "VALIDATE_SUB")
TAIL_PHASES = ("TELEGRAM_APPROVE", "SUBMIT", "FEEDBACK", "HEAL", "REPORT")


@dataclass
class CycleResult:
    competition: str
    dry_run: bool
    phases_run: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    context_sections: int = 0
    experiment_id: str | None = None
    plan_text: str | None = None
    kaggle_ok: bool | None = None
    browser_ok: bool | None = None
    deep_ok: bool | None = None
    deep_learnings: int = 0
    deep_sources: int = 0
    code_ok: bool | None = None
    smoke_ok: bool | None = None
    smoke_path: str | None = None
    kernel_ok: bool | None = None
    kernel_ref: str | None = None
    kernel_path: str | None = None
    validate_ok: bool | None = None
    candidate_csv: str | None = None
    approve_ok: bool | None = None
    submit_ok: bool | None = None
    submit_message: str | None = None
    waiting_approve: bool = False  # live run OK, needs /yes (not a failure)
    feedback_score: str | None = None
    heal_decision: str | None = None
    kernel_resumed: bool | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def hard_errors(self) -> list[str]:
        """Errors that mean the cycle failed (not waiting on human approve)."""
        return [e for e in self.errors if "need /approve" not in e and "need /yes" not in e]


def _parse_plan_lines(text: str) -> tuple[str, str]:
    hypothesis, approach = DEFAULT_HYPOTHESIS, "baseline"
    for line in text.splitlines():
        low = line.lower().strip()
        if low.startswith("hypothesis:"):
            hypothesis = line.split(":", 1)[1].strip()
        elif low.startswith("approach:"):
            approach = line.split(":", 1)[1].strip()
    return hypothesis, approach


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        competition: CompetitionConfig,
        root: Path | None = None,
        *,
        kaggle: KaggleClient | None = None,
        browser_fetch: FetchFn | None = None,
        telegram: SupportsTelegram | None = None,
        browser_submit: BrowserSubmitFn | None = None,
        mcp_submit_fn: Any | None = None,
        router: Any | None = None,
    ) -> None:
        self.settings = settings
        self.competition = competition
        self.root = root if root is not None else settings.root
        self.router = router if router is not None else ModelRouter.build(settings, competition)
        self._kaggle = kaggle
        self._browser_fetch = browser_fetch
        self._telegram = telegram
        self._browser_submit = browser_submit
        self._mcp_submit_fn = mcp_submit_fn  # tests inject call_tool

    def run_cycle(self, *, dry_run: bool | None = None) -> CycleResult:
        dry = self.settings.dry_run if dry_run is None else dry_run
        result = CycleResult(competition=self.competition.id, dry_run=dry)
        now = datetime.now(timezone.utc)
        state = load_state(self.root)

        if state.paused:
            return self._skip(result, "paused", now)
        lock = RunLock(self.root)
        if not lock.acquire():
            return self._skip(result, "lock_held", now)

        try:
            state = self._begin(state, dry, now, result)
            result.context_sections = len(build_context_pack(self.root).sections)

            # Block order is fixed (RESEARCH before the slice). Each block
            # is intersected with settings.phases so yaml can still drop steps.
            state = self._run_named_phases(
                self._enabled_phases(("LOCK",)), state, dry, result
            )
            state = self._run_named_phases(
                self._enabled_phases(("RESEARCH",)), state, dry, result
            )
            state = self._train_slice(state, dry, result)
            state = self._run_named_phases(
                self._enabled_phases(TAIL_PHASES), state, dry, result
            )

            self._finish_ok(state, result)
        except Exception as exc:  # noqa: BLE001
            self._finish_error(result, exc)
        finally:
            lock.release()
        return result

    def _skip(self, result: CycleResult, reason: str, now: datetime) -> CycleResult:
        result.skipped = True
        result.skip_reason = reason
        append_daily_log(f"skipped: {reason}", self.root, when=now)
        return result

    def _begin(
        self, state: AgentState, dry: bool, now: datetime, result: CycleResult
    ) -> AgentState:
        exp_id = now.strftime("%Y%m%d-%H%M%S") + ("-dry" if dry else "")
        result.experiment_id = exp_id
        state.lock_held = True
        state.dry_run = dry
        state.competition = self.competition.id
        state.active_experiment = exp_id
        state.last_cycle_start = now.isoformat()
        state.last_result = "running"
        state.last_error = "none"
        state.note = "running"
        save_state(state, self.root)
        append_daily_log(f"start {self.competition.id} dry={dry}", self.root, when=now)
        return state

    def _finish_ok(self, state: AgentState, result: CycleResult) -> None:
        state.phase = "IDLE"
        state.lock_held = False
        state.last_cycle_end = datetime.now(timezone.utc).isoformat()
        if result.hard_errors:
            state.last_result = "error"
        elif result.waiting_approve:
            state.last_result = "waiting_approve"
        else:
            state.last_result = "ok"
        if state.note in {"none", "running", "done", ""}:
            state.note = (
                "waiting /yes"
                if result.waiting_approve
                else "done"
            )
        save_state(state, self.root)
        if result.hard_errors:
            msg = f"end errors={result.hard_errors}"
        elif result.waiting_approve:
            msg = "end waiting_approve"
        else:
            msg = "end ok"
        append_daily_log(msg, self.root)

    def _finish_error(self, result: CycleResult, exc: Exception) -> None:
        result.errors.append(str(exc))
        state = load_state(self.root)
        state.phase = "IDLE"
        state.lock_held = False
        state.last_result = "error"
        state.last_error = str(exc)[:200]
        state.note = "error"
        save_state(state, self.root)
        append_daily_log(f"error: {exc}", self.root)

    def _enabled_phases(self, phases: tuple[str, ...]) -> tuple[str, ...]:
        allowed = set(self.settings.phases)
        return tuple(p for p in phases if p in allowed)

    def _run_named_phases(
        self,
        phases: tuple[str, ...] | list[str],
        state: AgentState,
        dry: bool,
        result: CycleResult,
    ) -> AgentState:
        for phase in phases:
            state.phase = phase
            save_state(state, self.root)
            append_daily_log(phase, self.root)
            state = self._phase(phase, state=state, dry=dry, result=result)
        return state

    def _train_slice(
        self, state: AgentState, dry: bool, result: CycleResult
    ) -> AgentState:
        return self._run_named_phases(
            self._enabled_phases(TRAIN_SLICE_PHASES), state, dry, result
        )

    def _phase(
        self,
        phase: str,
        *,
        state: AgentState,
        dry: bool,
        result: CycleResult,
    ) -> AgentState:
        result.phases_run.append(phase)
        handlers = {
            "RESEARCH": lambda: self._research(state, result),
            "PLAN": lambda: self._plan(state, dry, result) or state,
            "CODE": lambda: self._code(state, result),
            "LOCAL_SMOKE": lambda: self._local_smoke(state, result),
            "KERNEL_TRAIN": lambda: self._kernel_train(state, dry, result),
            "VALIDATE_SUB": lambda: self._validate_sub(state, result),
            "TELEGRAM_APPROVE": lambda: self._telegram_approve(state, dry, result),
            "SUBMIT": lambda: self._submit(state, dry, result),
            "FEEDBACK": lambda: self._feedback(state, dry, result),
            "REPORT": lambda: self._report(state, dry, result),
            "HEAL": lambda: self._heal(state, result),
        }
        if phase in handlers:
            return handlers[phase]()
        return state

    def _merge_budget(self, state: AgentState, updated: AgentState) -> None:
        state.budget_date = updated.budget_date
        state.max_proposals = updated.max_proposals
        state.proposals_used = updated.proposals_used
        state.note = updated.note

    def _research(self, state: AgentState, result: CycleResult) -> AgentState:
        try:
            if self._kaggle is None:
                self._kaggle = KaggleClient().connect()
            snap = self._kaggle.research_snapshot(self.competition.slug)
            updated = apply_kaggle_research(
                snap,
                self.root,
                agent_max_proposals=self.settings.max_proposals_per_day,
            )
            self._merge_budget(state, updated)
            result.kaggle_ok = not snap.errors
            if snap.errors:
                result.errors.extend(f"research:{e}" for e in snap.errors)
            allowed = snap.limits.num_allowed_now if snap.limits else "?"
            append_daily_log(
                f"kaggle research files={len(snap.meta_files)} lb={len(snap.leaderboard)} "
                f"kernels={len(snap.kernels)} allowed={allowed}",
                self.root,
            )
        except Exception as exc:  # noqa: BLE001
            result.kaggle_ok = False
            result.errors.append(f"research: {exc}")
            append_daily_log(f"kaggle research failed: {exc}", self.root)

        if self.settings.browser_research_enabled:
            self._browser_research(result)
        self._source_cards(result)
        self._deep_research(result)
        return state

    def _browser_research(self, result: CycleResult) -> None:
        try:
            researcher = (
                BrowserResearcher(fetch=self._browser_fetch)
                if self._browser_fetch is not None
                else BrowserResearcher.default(
                    prefer_browser_harness=self.settings.browser_prefer_harness
                )
            )
            notes = researcher.collect(
                self.competition.slug,
                pages=tuple(self.settings.browser_pages),
            )
            merge_browser_into_research_md(memory_dir(self.root) / "research.md", notes)
            result.browser_ok = bool(notes.pages) and not notes.errors
            append_daily_log(
                f"browser pages={list(notes.pages)} errors={notes.errors or 'none'}",
                self.root,
            )
        except Exception as exc:  # noqa: BLE001
            result.browser_ok = False
            append_daily_log(f"browser research failed: {exc}", self.root)

    def _source_cards(self, result: CycleResult) -> None:
        """One method card per top public kernel (in-process research workers)."""
        if self._kaggle is None:
            return
        try:
            our = str(load_state(self.root).public_best or "unknown")
            cache = self.root / self.competition.workspace_relative / "research-cache"
            cards = run_source_card_research(
                client=self._kaggle,
                competition=self.competition.slug,
                cache_dir=cache,
                root=self.root,
                our_score=our,
                log=lambda msg: append_daily_log(msg, self.root),
            )
            if cards:
                sidecar = write_methods_sidecar(
                    cards, self.root / self.competition.workspace_relative
                )
                append_daily_log(
                    f"source cards n={len(cards)} sidecar={sidecar.name}",
                    self.root,
                )
                result.deep_ok = True
                result.deep_sources = max(result.deep_sources, len(cards))
        except Exception as exc:  # noqa: BLE001
            append_daily_log(f"source cards failed: {exc}", self.root)

    def _deep_research(self, result: CycleResult) -> None:
        """Deep-research stage: recursive search over web/papers/notebooks/repos."""
        deep = self.settings.deep_research_config()
        if not deep.enabled:
            return
        try:
            if self._kaggle is None:
                self._kaggle = KaggleClient().connect()
            zen = self.router.client if self.router is not None else None
            if zen is None:
                append_daily_log("deep research skipped: no OPENCODE_API_KEY", self.root)
                return
            model = self.competition.model_for("distill", self.settings)
            cache = self.root / self.competition.workspace_relative / "research-cache"
            prompt = self._deep_prompt()
            researcher = DeepResearcher(
                zen,
                model,
                deep,
                sources=[
                    KaggleSource(self._kaggle, self.competition.slug, cache),
                    ArxivSource(),
                    GithubSource(),
                    WebSource(),
                ],
                root=self.root,
                log=lambda msg: append_daily_log(msg, self.root),
            )
            out = researcher.run(prompt, memory_dir(self.root) / "research.md")
            result.deep_ok = bool(out.learnings) and not out.error
            result.deep_learnings = len(out.learnings)
            result.deep_sources = len(out.sources)
            if out.error:
                result.errors.append(f"deep research: {out.error}")
            append_daily_log(
                f"deep research learnings={len(out.learnings)} sources={len(out.sources)} "
                f"queries={out.queries_run} error={out.error or 'none'}",
                self.root,
            )
        except Exception as exc:  # noqa: BLE001
            result.deep_ok = False
            append_daily_log(f"deep research failed: {exc}", self.root)

    def _deep_prompt(self) -> str:
        research_path = memory_dir(self.root) / "research.md"
        digest = ""
        if research_path.is_file():
            digest = research_path.read_text(encoding="utf-8")[:4000]
        labels = ", ".join(self.competition.labels)
        metric = self.competition.raw.get("metric", {}) or {}
        title = self.competition.raw.get("title", "")
        return (
            f"Competition: {self.competition.slug} ({title}). "
            f"Metric: {metric.get('name', '')} ({self.competition.metric_direction}). "
            f"Labels: {labels}. Submit mode: {self.competition.submit_mode}. "
            "Search ONLY this slug and its public kernels (site:kaggle.com/code), "
            "plus papers those notebooks actually cite. Ignore leaderboard-name "
            "collisions and off-topic arXiv. "
            "Must cover: official metric and submission header, data mount paths, "
            "top public notebooks, label sources, CV leakage, how test IDs are found. "
            "Return implementable steps the coding agent can copy: datasets to attach, "
            "inference ID discovery, ensemble rule, claimed public scores.\n\n"
            f"Current knowledge:\n{digest}"
        )

    def _code(self, state: AgentState, result: CycleResult) -> AgentState:
        workspace = self.root / self.competition.workspace_relative
        check = ensure_pipeline_ready(workspace)
        if check.ok:
            try:
                import sys

                sys.path.insert(0, str(workspace))
                from pipeline.recipe import apply_from_cards, apply_recipe  # type: ignore

                applied = apply_recipe(workspace, data_dir=self.root / "data")
                cards = apply_from_cards(workspace)
                result.code_ok = True
                append_daily_log(
                    f"code recipe {applied.message} cards={cards.message} "
                    f"present={check.present}",
                    self.root,
                )
                if not applied.ok:
                    append_daily_log(
                        f"code recipe skipped (kernel will train on Kaggle): {applied.message}",
                        self.root,
                    )
                if (
                    self.router is not None
                    and self.router.available()
                    and hasattr(self.router, "code")
                ):
                    try:
                        pack = build_context_pack(self.root)
                        brief = self.router.code(
                            CODE_SYSTEM,
                            f"Competition: {self.competition.slug}\n\n"
                            f"{pack.as_prompt_block()}",
                        )
                        brief_path = workspace / "pipeline" / "code_brief.md"
                        brief_path.write_text(brief, encoding="utf-8")
                        append_daily_log("code agent brief written", self.root)
                    except Exception as exc:  # noqa: BLE001
                        append_daily_log(f"code agent brief skipped: {exc}", self.root)
            except Exception as exc:  # noqa: BLE001
                result.code_ok = False
                result.errors.append(f"code: recipe failed: {exc}")
                append_daily_log(f"code recipe failed: {exc}", self.root)
        else:
            result.code_ok = False
            result.errors.append(f"code: missing {check.missing}")
            append_daily_log(f"code missing={check.missing}", self.root)
        return state

    def _local_smoke(self, state: AgentState, result: CycleResult) -> AgentState:
        sample_csv = _first_existing(
            self.root / "data" / "sample_submission.csv",
            self.root
            / "competitions"
            / self.competition.id
            / "data"
            / "sample_submission.csv",
        )
        outcome = run_competition_smoke(
            self.competition,
            root=self.root,
            exp_id=result.experiment_id or "smoke",
            sample_csv=sample_csv,
        )
        result.smoke_ok = outcome.ok
        if outcome.smoke and outcome.smoke.submission_path:
            result.smoke_path = str(outcome.smoke.submission_path)

        info = [e for e in outcome.errors if e.startswith("info:")]
        hard = [e for e in outcome.errors if not e.startswith("info:")]
        if not outcome.ok:
            result.errors.extend(f"smoke:{e}" for e in hard)
            append_daily_log(f"smoke failed: {outcome.errors}", self.root)
            return state

        n = outcome.smoke.n_studies if outcome.smoke else 0
        append_daily_log(
            f"smoke ok path={result.smoke_path} n={n} info={info or 'none'}",
            self.root,
        )
        # Do not promote info notes to hard cycle errors
        return state

    def _kernel_username(self) -> str:
        if self.settings.kernel_username:
            return self.settings.kernel_username
        if self._kaggle is not None:
            try:
                return self._kaggle.username()
            except Exception:  # noqa: BLE001
                pass
        # Offline / dry package: placeholder is fine until push
        return "local-user"

    def _kernel_train(
        self, state: AgentState, dry: bool, result: CycleResult
    ) -> AgentState:
        exp_id = result.experiment_id or "kernel"
        should_push = self.settings.kernel_push and not dry
        existing = load_kernel_job(self.root)

        try:
            package = None
            # Resume in-flight job: do not invent a new notebook push
            if should_push and existing.is_active:
                result.kernel_ref = existing.kernel_ref
                result.kernel_path = (
                    existing.folder if existing.folder not in {"none", ""} else None
                )
                if self._kaggle is None:
                    self._kaggle = KaggleClient().connect()
                out_dir = (
                    Path(existing.folder) / "output"
                    if existing.folder not in {"none", ""}
                    else None
                )
                run = run_kernel_phase(
                    self._kaggle,
                    None,
                    push=True,
                    pull_output_dir=out_dir,
                    root=self.root,
                    competition=self.competition.slug,
                    exp_id=exp_id,
                )
            else:
                package = write_kernel_package(
                    self.competition,
                    root=self.root,
                    username=self._kernel_username(),
                    exp_id=exp_id,
                    enable_gpu=self.settings.kernel_enable_gpu,
                )
                result.kernel_path = str(package.folder)
                result.kernel_ref = package.kernel_ref
                if should_push and self._kaggle is None:
                    self._kaggle = KaggleClient().connect()
                out_dir = package.folder / "output"
                run = run_kernel_phase(
                    self._kaggle if should_push else None,
                    package,
                    push=should_push,
                    pull_output_dir=out_dir if should_push else None,
                    root=self.root,
                    competition=self.competition.slug,
                    exp_id=exp_id,
                )

            result.kernel_ok = run.ok
            result.kernel_resumed = run.resumed
            if run.kernel_ref and run.kernel_ref != "none":
                result.kernel_ref = run.kernel_ref
            if not run.ok:
                result.errors.extend(f"kernel:{e}" for e in run.errors)
            append_daily_log(
                f"kernel ok={run.ok} push={run.pushed} resume={run.resumed} "
                f"status={run.status} ref={result.kernel_ref} path={result.kernel_path}",
                self.root,
            )
        except Exception as exc:  # noqa: BLE001
            result.kernel_ok = False
            result.errors.append(f"kernel: {exc}")
            append_daily_log(f"kernel failed: {exc}", self.root)
        return state

    def _validate_sub(self, state: AgentState, result: CycleResult) -> AgentState:
        """Validate best local candidate CSV (kernel output preferred, else smoke)."""
        candidates: list[Path] = []
        if result.kernel_path:
            candidates.append(Path(result.kernel_path) / "output" / "submission.csv")
        if result.smoke_path:
            candidates.append(Path(result.smoke_path))
        # Also check latest smoke under submissions/
        sub_dir = self.root / self.competition.workspace_relative / "submissions"
        if sub_dir.is_dir():
            candidates.extend(sorted(sub_dir.glob("*_smoke.csv"), reverse=True)[:1])

        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            result.validate_ok = False
            result.errors.append("validate: no candidate CSV")
            append_daily_log("validate: no candidate CSV", self.root)
            return state

        check = validate_submission_csv(
            path,
            id_column=self.competition.id_column,
            labels=self.competition.labels,
        )
        result.validate_ok = check.ok
        result.candidate_csv = str(path)
        if check.ok:
            append_daily_log(f"validate ok path={path} rows={check.n_rows}", self.root)
        else:
            result.errors.extend(f"validate:{e}" for e in check.errors[:5])
            append_daily_log(f"validate failed: {check.errors[:3]}", self.root)
        return state

    def _plan(self, state: AgentState, dry: bool, result: CycleResult) -> None:
        hypothesis, approach, notes = DEFAULT_HYPOTHESIS, "baseline", "no LLM"
        pack = build_context_pack(self.root)

        if self.router.available():
            try:
                text = self.router.plan(
                    PLAN_SYSTEM,
                    f"Competition: {self.competition.slug}\n\n{pack.as_prompt_block()}",
                )
                result.plan_text = text
                hypothesis, approach = _parse_plan_lines(text)
                notes = "zen plan"
            except Exception as exc:  # noqa: BLE001
                notes = f"zen plan failed: {exc}"
                result.errors.append(f"plan: {exc}")
        else:
            notes = "OPENCODE_API_KEY missing; offline plan"

        write_experiment(
            result.experiment_id or "unknown",
            hypothesis=hypothesis,
            approach=approach,
            notes=notes + ("; dry_run" if dry else ""),
            root=self.root,
        )

    def _tg(self) -> SupportsTelegram | None:
        if self._telegram is not None:
            return self._telegram
        if self.settings.telegram_enabled:
            self._telegram = TelegramClient.from_env()
        return self._telegram

    def _notify(self, text: str) -> None:
        client = self._tg()
        if client is None:
            return
        try:
            client.send_message(text)
        except Exception as exc:  # noqa: BLE001 — notify never kills cycle
            append_daily_log(f"telegram send failed: {exc}", self.root)

    def _telegram_approve(
        self, state: AgentState, dry: bool, result: CycleResult
    ) -> AgentState:
        exp_id = result.experiment_id or state.active_experiment
        csv_path = result.candidate_csv or "none"
        if not result.validate_ok or not result.candidate_csv:
            result.approve_ok = False
            append_daily_log("approve skipped: no validated candidate", self.root)
            return state

        # Live: keep prior /yes so a second /run live can submit without re-asking
        if not dry:
            prior = usable_approval(self.root, competition=self.competition.slug)
            if prior is not None:
                result.approve_ok = True
                result.candidate_csv = prior.csv_path
                state.pending_approve = prior.exp_id
                append_daily_log(
                    f"approve reuse exp={prior.exp_id} csv={prior.csv_path}",
                    self.root,
                )
                self._notify(
                    "Using your previous /yes approval.\n\n"
                    f"Approved experiment: {prior.exp_id}\n"
                    f"CSV: {prior.csv_path}\n"
                    f"This cycle: {exp_id}\n\n"
                    "Submit will use the approved CSV next."
                )
                return state

        pending = request_approval(
            exp_id=exp_id,
            csv_path=csv_path,
            competition=self.competition.slug,
            message="awaiting /approve" if not dry else "dry-run request",
            root=self.root,
            kernel_path=result.kernel_path or "none",
            kernel_ref=result.kernel_ref or "none",
        )
        state.pending_approve = exp_id
        mode = "dry run (no real submit)" if dry else "live run"
        msg = (
            "Approval needed\n\n"
            f"Mode: {mode}\n"
            f"Experiment: {exp_id}\n"
            f"Competition: {self.competition.slug}\n"
            f"CSV: {csv_path}\n\n"
            "To allow submit: /yes\n"
            "To block: /no\n"
            f"Or: /approve {exp_id}"
        )
        self._notify(msg)
        if dry:
            result.approve_ok = True
            append_daily_log(f"approve requested (dry) exp={exp_id}", self.root)
        else:
            result.approve_ok = pending.status == "pending"
            result.waiting_approve = True
            append_daily_log(f"approve pending exp={exp_id}", self.root)
        return state

    def _submit(self, state: AgentState, dry: bool, result: CycleResult) -> AgentState:
        csv_path = result.candidate_csv
        if not csv_path or not Path(csv_path).is_file():
            result.submit_ok = False
            result.errors.append("submit: no candidate csv")
            append_daily_log("submit skipped: no csv", self.root)
            return state

        pending = load_pending(self.root)
        if not dry and self.settings.require_telegram_approve:
            if (
                pending.status == "approved"
                and pending.csv_path not in {csv_path, "none"}
                and Path(pending.csv_path).is_file()
            ):
                csv_path = pending.csv_path
                result.candidate_csv = csv_path
            if pending.status != "approved":
                result.submit_ok = False
                result.waiting_approve = True
                append_daily_log(
                    f"submit waiting_approve status={pending.status}", self.root
                )
                self._notify(
                    "Live cycle ready — waiting for your approval.\n\n"
                    f"Experiment: {result.experiment_id}\n"
                    f"CSV: {csv_path}\n"
                    f"Status: {pending.status}\n\n"
                    "Everything else succeeded.\n\n"
                    "Send /yes now, then /run live again.\n"
                    "After /yes, the next live run will submit without asking again."
                )
                return state

        mode = self.competition.submit_mode
        kernel_folder: Path | None = None
        kernel_ref = normalize_kernel_ref(result.kernel_ref)
        for path_str in (pending.kernel_path, result.kernel_path or ""):
            if path_str not in {"", "none"} and Path(path_str).is_dir():
                kernel_folder = Path(path_str)
                break
        if pending.kernel_ref not in {"", "none"}:
            kernel_ref = normalize_kernel_ref(pending.kernel_ref)
        if kernel_folder and pending.kernel_path in {"", "none"}:
            pending.kernel_path = str(kernel_folder)
        if kernel_ref and pending.kernel_ref in {"", "none"}:
            pending.kernel_ref = kernel_ref
        if pending.status == "approved":
            save_pending(pending, self.root)

        msg = f"agent {result.experiment_id}" + (" dry" if dry else "")
        fails: list[str] = []
        result.submit_ok = False
        result.submit_message = ""

        # --- 1) Kaggle MCP ---
        if self.settings.mcp_submit:
            append_daily_log(f"submit mcp start mode={mode} kernel={kernel_ref}", self.root)
            if not dry:
                self._notify(
                    f"Submitting via Kaggle MCP ({mode})…\n"
                    f"Experiment: {result.experiment_id}"
                )
            try:
                sr = submit_via_mcp(
                    competition=self.competition.slug,
                    message=msg,
                    mode=mode,
                    csv_path=Path(csv_path),
                    kernel_ref=kernel_ref,
                    output_file=self.competition.submit_output_file,
                    dry_run=dry,
                    call_tool=self._mcp_submit_fn,
                )
                result.submit_ok = sr.success
                result.submit_message = f"mcp: {sr.message}"
                if not sr.success:
                    fails.append(f"mcp: {sr.message}")
            except Exception as exc:  # noqa: BLE001
                fails.append(f"mcp: {exc}")
                result.submit_message = f"mcp: {exc}"
                append_daily_log(f"submit mcp failed: {exc}", self.root)

        # --- 2) Python API ---
        if not result.submit_ok and self.settings.api_submit:
            append_daily_log(f"submit api start mode={mode} kernel={kernel_ref}", self.root)
            try:
                if dry and self._kaggle is None and not self.settings.mcp_submit:
                    result.submit_ok = True
                    result.submit_message = (
                        f"dry_run: would {mode}-submit {csv_path} / kernel={kernel_ref}"
                    )
                else:
                    if self._kaggle is None and not dry:
                        self._kaggle = KaggleClient().connect()
                    if dry and self._kaggle is None:
                        result.submit_ok = True
                        result.submit_message = (
                            f"dry_run: would api {mode}-submit {csv_path} kernel={kernel_ref}"
                        )
                    else:
                        if not dry:
                            self._notify(
                                f"MCP failed — trying Kaggle API ({mode})…\n"
                                f"Last: {(fails[-1] if fails else '')[:200]}"
                            )
                        nb = mode == "notebook"
                        assert self._kaggle is not None
                        sr = self._kaggle.submit(
                            self.competition.slug,
                            Path(csv_path),
                            msg,
                            dry_run=dry,
                            mode=mode,
                            kernel_folder=kernel_folder,
                            kernel_ref=kernel_ref,
                            output_file=self.competition.submit_output_file,
                            poll_seconds=30 if nb else 15,
                            poll_attempts=40 if nb else 10,
                        )
                        result.submit_ok = sr.success
                        result.submit_message = f"api: {sr.message}"
                        if not sr.success:
                            fails.append(f"api: {sr.message}")
            except Exception as exc:  # noqa: BLE001
                fails.append(f"api: {exc}")
                result.submit_message = f"api: {exc}"
                append_daily_log(f"submit api failed: {exc}", self.root)

        # --- 3) Browser-harness ---
        if (
            not dry
            and not result.submit_ok
            and self.settings.browser_submit_fallback
        ):
            append_daily_log(
                f"submit browser after mcp/api: {fails[-1] if fails else ''}",
                self.root,
            )
            self._notify(
                "MCP and API failed — trying browser-harness…\n\n"
                f"Last error: {(fails[-1] if fails else 'unknown')[:300]}\n"
                "Chrome must already be signed in to Kaggle."
            )
            try:
                br = submit_via_browser(
                    BrowserSubmitRequest(
                        competition=self.competition.slug,
                        message=msg,
                        mode=mode,
                        csv_path=Path(csv_path),
                        kernel_ref=kernel_ref,
                        dry_run=False,
                    ),
                    run_fn=self._browser_submit,
                )
                result.submit_ok = br.success
                result.submit_message = f"browser: {br.message}"
                if not br.success:
                    fails.append(f"browser: {br.message}")
            except Exception as exc:  # noqa: BLE001
                fails.append(f"browser: {exc}")
                result.submit_message = f"browser: {exc}"
                append_daily_log(f"submit browser failed: {exc}", self.root)

        if not dry and result.submit_ok:
            mark_submitted(self.root)
            try:
                used = int(state.proposals_used or "0") + 1
            except ValueError:
                used = 1
            state.proposals_used = str(used)
            state.pending_approve = "none"
        elif not dry and not result.submit_ok:
            for f in fails[:5]:
                result.errors.append(f"submit {f}")

        kind = "dry" if dry else "real"
        append_daily_log(
            f"submit {kind} mode={mode}: {result.submit_message}", self.root
        )
        if not dry:
            self._notify(
                ("Submitted to Kaggle.\n\n" if result.submit_ok else "Submit failed.\n\n")
                + f"Mode: {mode}\n"
                f"Experiment: {result.experiment_id}\n"
                f"Result: {result.submit_message}"
            )
        return state

    def _feedback(self, state: AgentState, dry: bool, result: CycleResult) -> AgentState:
        if dry or not result.submit_ok:
            append_daily_log("feedback skipped", self.root)
            return state
        try:
            if self._kaggle is None:
                self._kaggle = KaggleClient().connect()
            subs = self._kaggle.submissions(self.competition.slug, top=3)
            if subs:
                latest = subs[0]
                result.feedback_score = latest.public_score or latest.status
                append_daily_log(
                    f"feedback status={latest.status} score={latest.public_score}",
                    self.root,
                )
                if latest.public_score and latest.public_score not in {"", "none"}:
                    state.public_best = latest.public_score
            else:
                append_daily_log("feedback: no submissions yet", self.root)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"feedback: {exc}")
            append_daily_log(f"feedback failed: {exc}", self.root)
        return state

    def _heal(self, state: AgentState, result: CycleResult) -> AgentState:
        heal = load_heal(self.root)
        score = result.feedback_score
        # Prefer public LB score; fall back to personal best already stored
        if not score or score in {"none", "n/a"}:
            score = None

        cycle_ok = not any(
            e.startswith(("code:", "smoke:", "kernel:", "validate:"))
            for e in result.errors
        )
        if result.validate_ok and not result.submit_ok and not result.dry_run:
            # Waiting on human approve is not a model failure
            heal.decision_next = "wait_approve"
            heal.note = "submit blocked — need /approve"
            save_heal(heal, self.root)
            result.heal_decision = heal.decision_next
            append_daily_log(f"heal decision_next={heal.decision_next}", self.root)
            return state

        heal = decide_next(
            heal,
            public_score=score,
            metric_direction=self.competition.metric_direction,
            max_tune_attempts=self.settings.max_tune_attempts,
            max_no_improve_days=self.settings.max_no_improve_days,
            cycle_ok=cycle_ok,
        )
        save_heal(heal, self.root)
        result.heal_decision = heal.decision_next
        append_daily_log(
            f"heal decision_next={heal.decision_next} note={heal.note} "
            f"flat_days={heal.no_improve_days} best={heal.best_score}",
            self.root,
        )

        if heal.decision_next == "pause":
            state.paused = True
            state.note = f"heal pause: {heal.note}"
            self._notify(
                "Agent paused by heal policy.\n\n"
                f"Reason: {heal.note}\n"
                f"Best public score so far: {heal.best_score}\n"
                f"Days without improvement: {heal.no_improve_days}\n\n"
                "Send /resume when you want it to continue."
            )
        return state

    def _report(self, state: AgentState, dry: bool, result: CycleResult) -> AgentState:
        mode = "dry run (safe)" if dry else "live run"

        def _flag(label: str, ok: bool | None) -> str:
            if ok is True:
                return f"✓ {label}"
            if ok is False:
                return f"✗ {label}"
            return f"· {label} (n/a)"

        steps = "\n".join(
            [
                _flag("Kaggle research", result.kaggle_ok),
                _flag("Browser research", result.browser_ok),
                _flag("Code workspace", result.code_ok),
                _flag("Local smoke", result.smoke_ok),
                _flag("Kernel package", result.kernel_ok),
                _flag("CSV validate", result.validate_ok),
                _flag("Approve step", result.approve_ok),
                _flag("Submit", result.submit_ok),
            ]
        )
        hard = result.hard_errors
        err_block = (
            "\n".join(f"• {e[:160]}" for e in hard[:5]) if hard else "None"
        )
        if result.waiting_approve:
            headline = "Cycle finished — waiting for your approval"
            next_step = (
                "Next step (required):\n"
                "1) Send /yes to approve this candidate\n"
                "2) Send /run live so it can submit to Kaggle\n"
            )
        elif hard:
            headline = f"Cycle finished with problems — {mode}"
            next_step = "Next: check Errors below, then /status or /run again.\n"
        else:
            headline = f"Cycle finished — {mode}"
            next_step = "Next: /status · /run · /help\n"

        text = (
            f"{headline}\n\n"
            f"Experiment: {result.experiment_id}\n"
            f"Competition: {self.competition.slug}\n\n"
            f"Steps\n{steps}\n\n"
            f"Score / feedback: {result.feedback_score or 'none yet'}\n"
            f"Heal next: {result.heal_decision or 'n/a'}\n"
            f"Kernel resumed existing job: {result.kernel_resumed}\n\n"
            f"Candidate CSV:\n{result.candidate_csv or 'none'}\n\n"
            f"Kernel package:\n{result.kernel_path or 'none'}\n"
            f"Kernel ref: {result.kernel_ref or 'none'}\n\n"
            f"Hard errors:\n{err_block}\n\n"
            f"{next_step}"
        )
        append_daily_log("report sent", self.root)
        self._notify(text)
        return state


def run_daily(
    competition_id: str | None = None,
    *,
    root: Path | None = None,
    dry_run: bool | None = None,
    kaggle: KaggleClient | None = None,
    browser_fetch: FetchFn | None = None,
    telegram: SupportsTelegram | None = None,
    browser_submit: BrowserSubmitFn | None = None,
    mcp_submit_fn: Any | None = None,
) -> CycleResult:
    settings = load_settings(root)
    cid = competition_id or settings.default_competition
    competition = load_competition(cid, root or settings.root)
    return Orchestrator(
        settings,
        competition,
        root=root or settings.root,
        kaggle=kaggle,
        browser_fetch=browser_fetch,
        telegram=telegram,
        browser_submit=browser_submit,
        mcp_submit_fn=mcp_submit_fn,
    ).run_cycle(dry_run=dry_run)

