"""Daily cycle: Kaggle research, browser research, pipeline smoke, optional Zen PLAN."""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_agent.code.workspace import ensure_pipeline_ready
from kaggle_agent.config import CompetitionConfig, Settings, load_competition, load_settings
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.kaggle_api.models import SubmissionRow
from kaggle_agent.judge import (
    judge_kernel,
    judge_plan,
    judge_train_llm,
    new_judge_state,
)
from kaggle_agent.llm.router import ModelRouter
from kaggle_agent.memory.ingest import build_context_pack
from kaggle_agent.memory.write import (
    append_daily_log,
    patch_experiment,
    patch_memory_public_score,
    write_experiment,
)
from kaggle_agent.notify.telegram import SupportsTelegram, TelegramClient
from kaggle_agent.paths import memory_dir
from kaggle_agent.pipeline.validate import validate_submission_csv
from kaggle_agent.research.apply_snapshot import apply_kaggle_research
from kaggle_agent.research.browser import (
    BrowserResearcher,
    FetchFn,
    default_fetch,
    default_serp,
    merge_browser_into_research_md,
)
from kaggle_agent.agents.code import make_code_agent
from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.agents.plan import make_plan_agent, write_plan_text
from kaggle_agent.agents.verification import (
    verify_code_stage,
    verify_plan_stage,
    verify_research_fleet,
)
from kaggle_agent.research.agent import ResearchAgent
from kaggle_agent.research.fleet import (
    AGENT_SPECS,
    clone_client_for_agent,
    fleet_tool_schemas,
    make_fleet_tools,
    make_write_card,
    run_fleet,
    subagent_system,
)
from kaggle_agent.research.source_cards import (
    _PULL_LOCK,
    cards_feasible,
    judge_cards_ready,
    load_methods,
    merge_digest,
    run_source_card_research,
    write_methods_sidecar,
)
from kaggle_agent.stages import Stage, build_stage_registry
from kaggle_agent.state_access import DiskStateAccessor, StateAccessor
from kaggle_agent.research.deep import (
    ArxivSource,
    DatasetSource,
    DeepResearcher,
    DiscussionSource,
    GithubSource,
    KaggleSource,
    WebSource,
)
from kaggle_agent.state_md import AgentState
from kaggle_agent.kaggle_api.mcp_submit import submit_via_mcp
from kaggle_agent.kaggle_api.submit_ops import normalize_kernel_ref
from kaggle_agent.submit.browser_submit import (
    BrowserSubmitFn,
    BrowserSubmitRequest,
    submit_via_browser,
)
from kaggle_agent.submit.pending import (
    mark_submitted,
    request_approval,
    set_decision,
    usable_approval,
)
from kaggle_agent.heal.policy import decide_next, load_heal, save_heal
from kaggle_agent.heal.feedback import (
    already_recorded,
    exp_id_from_description,
)
from kaggle_agent.loop import (
    load_loop,
    next_loop_count,
    parse_loop_score,
    score_is_better,
    update_loop_from_score,
)
from kaggle_agent.experiment_fingerprint import submission_output_hash
from kaggle_agent.train.kernel_history import record_output, seen_output
from kaggle_agent.train.kernel_runner import (
    KernelRunResult,
    package_matches_existing,
    run_kernel_phase,
)
from kaggle_agent.train.local_smoke import run_competition_smoke
from kaggle_agent.train.notebook_builder import write_kernel_package
from kaggle_agent.ops.evals import evaluate_cycle, persist_report
from kaggle_agent.ops.tracing import Tracer

DEFAULT_HYPOTHESIS = "dry-run default: schema-valid 0.5 baseline then improve"

TRAIN_SLICE_PHASES = ("PLAN", "CODE", "LOCAL_SMOKE", "KERNEL_TRAIN", "VALIDATE_SUB")
SUBMIT_PHASES = ("TELEGRAM_APPROVE", "SUBMIT", "FEEDBACK")
TAIL_PHASES = ("HEAL", "REPORT")
_SLICE_ERR_PREFIXES = ("code:", "smoke:", "kernel:", "validate:")


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
    research_verified: bool | None = None
    research_verification_detail: str = ""
    deep_learnings: int = 0
    deep_sources: int = 0
    code_ok: bool | None = None
    smoke_ok: bool | None = None
    smoke_path: str | None = None
    kernel_ok: bool | None = None
    kernel_duplicate: bool = False
    output_duplicate: bool = False
    kernel_ref: str | None = None
    kernel_path: str | None = None
    kernel_judge_ok: bool | None = None
    validate_ok: bool | None = None
    candidate_csv: str | None = None
    approve_ok: bool | None = None
    submit_ok: bool | None = None
    submit_message: str | None = None
    waiting_approve: bool = False  # live run OK, needs /yes (not a failure)
    feedback_score: str | None = None
    heal_decision: str | None = None
    kernel_resumed: bool | None = None
    train_slices: int = 0
    research_passes: int = 0
    wrote_custom_infer: bool = False
    wrote_methods: bool = False
    wrote_recipe: bool = False
    plan_verified: bool | None = None
    code_verified: bool | None = None
    code_agent: str | None = None
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
        skip_phases: frozenset[str] | None = None,
        state_access: StateAccessor | None = None,
    ) -> None:
        self.settings = settings
        self.competition = competition
        self.root = root if root is not None else settings.root
        self.router = router if router is not None else ModelRouter.build(settings, competition)
        self._sa = state_access if state_access is not None else DiskStateAccessor(self.root)
        self._kaggle = kaggle
        self._browser_fetch = browser_fetch
        self._telegram = telegram
        self._browser_submit = browser_submit
        self._mcp_submit_fn = mcp_submit_fn  # tests inject call_tool
        self._assume_approved = False
        self._loop_n_used = 0
        self._tracer: Tracer | None = None
        self._skip_phases = skip_phases or frozenset()
        self._stages: dict[str, Stage] = build_stage_registry(self)

    def run_cycle(
        self, *, dry_run: bool | None = None, assume_approved: bool = False
    ) -> CycleResult:
        dry = self.settings.dry_run if dry_run is None else dry_run
        self._assume_approved = assume_approved
        self._loop_n_used = 0
        result = CycleResult(competition=self.competition.id, dry_run=dry)
        now = datetime.now(timezone.utc)
        state = self._sa.load_state()

        if state.paused:
            return self._skip(result, "paused", now)
        if not self._sa.acquire_lock():
            return self._skip(result, "lock_held", now)
        if self._sa.lock_took_over():
            append_daily_log("run lock stale; taken over", self.root)

        try:
            state = self._begin(state, dry, now, result)
            result.context_sections = len(build_context_pack(self.root).sections)

            if not dry:
                try:
                    self._catch_up_scores(state)
                except Exception as exc:  # noqa: BLE001
                    append_daily_log(f"score catch-up failed: {exc}", self.root)

            # Block order is fixed (RESEARCH before the slice). Each block
            # is intersected with settings.phases so yaml can still drop steps.
            state = self._run_named_phases(
                self._enabled_phases(("LOCK",)), state, dry, result
            )
            state = self._run_named_phases(
                self._enabled_phases(("RESEARCH",)), state, dry, result
            )
            research_blocked = result.research_verified is False and self._llm_available()
            if research_blocked:
                result.errors.append(
                    "research verification failed; training blocked: "
                    f"{result.research_verification_detail}"
                )
                append_daily_log("research verification failed; training blocked", self.root)
            else:
                state = self._run_train_slices(state, dry, result, started=now)
            state = self._run_named_phases(
                self._enabled_phases(SUBMIT_PHASES), state, dry, result
            )
            self._update_loop_after_feedback(result)
            state = self._run_named_phases(
                self._enabled_phases(TAIL_PHASES), state, dry, result
            )

            self._finish_ok(state, result)
        except Exception as exc:  # noqa: BLE001
            self._finish_error(result, exc)
        finally:
            self._sa.release_lock()
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
        self._tracer = Tracer(self.root, cycle_id=exp_id)
        self._tracer.emit("cycle_start", competition=self.competition.id, dry=dry)
        state.lock_held = True
        state.dry_run = dry
        state.competition = self.competition.id
        state.active_experiment = exp_id
        state.last_cycle_start = now.isoformat()
        state.last_result = "running"
        state.last_error = "none"
        state.note = "running"
        self._sa.save_state(state)
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
        self._sa.save_state(state)
        if result.hard_errors:
            msg = f"end errors={result.hard_errors}"
        elif result.waiting_approve:
            msg = "end waiting_approve"
        else:
            msg = "end ok"
        append_daily_log(msg, self.root)
        self._ops_close(result, state.last_result)

    def _finish_error(self, result: CycleResult, exc: Exception) -> None:
        result.errors.append(str(exc))
        state = self._sa.load_state()
        state.phase = "IDLE"
        state.lock_held = False
        state.last_result = "error"
        state.last_error = str(exc)[:200]
        state.note = "error"
        self._sa.save_state(state)
        append_daily_log(f"error: {exc}", self.root)
        self._ops_close(result, "error")

    def _ops_close(self, result: CycleResult, status: str) -> None:
        if self._tracer is not None:
            self._tracer.emit("cycle_end", status=status, errors=result.hard_errors[:5])
        try:
            persist_report(self.root, evaluate_cycle(self.root))
        except Exception:  # noqa: BLE001
            pass

    def _enabled_phases(self, phases: tuple[str, ...]) -> tuple[str, ...]:
        allowed = set(self.settings.phases)
        return tuple(p for p in phases if p in allowed and p not in self._skip_phases)

    def _llm_available(self) -> bool:
        client = self.router.client if self.router is not None else None
        return client is not None and hasattr(client, "chat")

    def _run_named_phases(
        self,
        phases: tuple[str, ...] | list[str],
        state: AgentState,
        dry: bool,
        result: CycleResult,
    ) -> AgentState:
        for phase in phases:
            state.phase = phase
            self._sa.save_state(state)
            append_daily_log(phase, self.root)
            if self._tracer is not None:
                self._tracer.emit("phase", phase=phase)
            state = self._phase(phase, state=state, dry=dry, result=result)
        return state

    def _resolve_loop_n(self) -> int:
        n_min = self.settings.loop_n_min
        n_max = self.settings.loop_n_max
        loop = load_loop(self.root)
        try:
            n = int(str(loop.next_n).strip())
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            n = next_loop_count(
                None,
                n_min=n_min,
                n_max=n_max,
                typical_gain=self.settings.loop_typical_gain,
                default_n=self.settings.loop_default_n,
            )
        else:
            # Honor explicit next_n=1 in tests; still cap at n_max.
            cap = n_max if n_max > 0 else n
            n = max(1, min(n, cap))
        return n

    def _clear_slice_fields(self, result: CycleResult) -> None:
        result.code_ok = None
        result.code_verified = None
        result.code_agent = None
        result.smoke_ok = None
        result.smoke_path = None
        result.kernel_ok = None
        result.kernel_duplicate = False
        result.output_duplicate = False
        result.kernel_ref = None
        result.kernel_path = None
        result.kernel_resumed = None
        result.validate_ok = None
        result.candidate_csv = None
        result.wrote_custom_infer = False
        result.wrote_methods = False
        result.wrote_recipe = False

    def _run_train_slices(
        self,
        state: AgentState,
        dry: bool,
        result: CycleResult,
        *,
        started: datetime,
    ) -> AgentState:
        n = self._resolve_loop_n()
        base_exp = result.experiment_id or "exp"
        best: dict[str, Any] | None = None
        used = 0
        for i in range(1, n + 1):
            if i > 1:
                if self._sa.load_state().paused:
                    append_daily_log("train loop stop: paused", self.root)
                    break
                limit = self.settings.loop_max_minutes
                if limit > 0:
                    elapsed = (
                        datetime.now(timezone.utc) - started
                    ).total_seconds() / 60.0
                    if elapsed >= limit:
                        append_daily_log("train loop stop: max_minutes", self.root)
                        break
            if n > 1:
                result.experiment_id = f"{base_exp}-s{i}"
                state.active_experiment = result.experiment_id
                self._sa.save_state(state)
            self._clear_slice_fields(result)
            err_before = len(result.errors)
            append_daily_log(f"train slice {i}/{n}", self.root)
            state = self._train_slice(state, dry, result)
            used = i
            csv = result.candidate_csv
            if result.validate_ok and csv and Path(csv).is_file():
                prior = [
                    e
                    for e in result.errors[:err_before]
                    if not e.startswith(_SLICE_ERR_PREFIXES)
                ]
                best = {
                    "experiment_id": result.experiment_id,
                    "kernel_path": result.kernel_path,
                    "kernel_ref": result.kernel_ref,
                    "candidate_csv": csv,
                    "smoke_path": result.smoke_path,
                    "kernel_ok": result.kernel_ok,
                    "wrote_custom_infer": result.wrote_custom_infer,
                    "wrote_methods": result.wrote_methods,
                    "wrote_recipe": result.wrote_recipe,
                    "errors": prior + result.errors[err_before:],
                }
        self._loop_n_used = used
        result.train_slices = used
        if best:
            result.experiment_id = best["experiment_id"]
            result.kernel_path = best["kernel_path"]
            result.kernel_ref = best["kernel_ref"]
            result.candidate_csv = best["candidate_csv"]
            result.smoke_path = best["smoke_path"]
            result.kernel_ok = best["kernel_ok"]
            result.wrote_custom_infer = bool(best.get("wrote_custom_infer"))
            result.wrote_methods = bool(best.get("wrote_methods"))
            result.wrote_recipe = bool(best.get("wrote_recipe"))
            result.validate_ok = True
            result.errors = list(best["errors"])
            state.active_experiment = result.experiment_id
        else:
            result.validate_ok = False
            result.candidate_csv = None
        return state

    def _update_loop_after_feedback(self, result: CycleResult) -> None:
        score = result.feedback_score
        if parse_loop_score(score) is None:
            return
        update_loop_from_score(
            self.root,
            score,
            n_used=self._loop_n_used or 1,
            n_min=self.settings.loop_n_min,
            n_max=self.settings.loop_n_max,
            typical_gain=self.settings.loop_typical_gain,
            default_n=self.settings.loop_default_n,
            direction=self.competition.metric_direction,
        )

    def _train_slice(
        self, state: AgentState, dry: bool, result: CycleResult
    ) -> AgentState:
        for phase in self._enabled_phases(TRAIN_SLICE_PHASES):
            state = self._run_named_phases((phase,), state, dry, result)
            if phase == "PLAN" and result.plan_verified is False and self._llm_available():
                result.errors.append("plan verification failed; code blocked")
                append_daily_log("plan verification failed; code blocked", self.root)
                break
            if phase == "CODE" and result.code_ok is False:
                append_daily_log("train slice stopped: CODE produced no recipe change", self.root)
                break
            if phase == "LOCAL_SMOKE" and result.code_verified is False:
                append_daily_log("code verification failed; train slice stopped", self.root)
                break
            if phase == "KERNEL_TRAIN" and result.kernel_duplicate:
                append_daily_log("train slice stopped: kernel identical to a previous run", self.root)
                break
            if phase == "VALIDATE_SUB" and result.output_duplicate:
                append_daily_log("train slice stopped: output identical to a previous run", self.root)
                break
        return state

    def _phase(
        self,
        phase: str,
        *,
        state: AgentState,
        dry: bool,
        result: CycleResult,
    ) -> AgentState:
        result.phases_run.append(phase)
        stage = self._stages.get(phase)
        if stage is None:
            return state
        return stage.run(state, dry, result) or state

    def _merge_budget(self, state: AgentState, updated: AgentState) -> None:
        state.budget_date = updated.budget_date
        state.max_proposals = updated.max_proposals
        state.proposals_used = updated.proposals_used
        state.note = updated.note

    def _research(self, state: AgentState, result: CycleResult) -> AgentState:
        workspace = self.root / self.competition.workspace_relative
        research_md = memory_dir(self.root) / "research.md"
        self._kaggle_snapshot(state, result)
        if self.settings.browser_research_enabled:
            self._browser_research(result)
        zen = self.router.client if self.router is not None else None
        model = self.competition.model_for("distill", self.settings)
        if self._fleet_enabled() and self._fleet_roster():
            self._fleet_research(state, result)
            if self.settings.deep_research_config().enabled:
                self._deep_research(result)
            return state
        thin = not cards_feasible(workspace, research_md)
        deep_ran = {"n": 0}
        dest = memory_dir(self.root) / "research-deep"
        our = str(self._sa.load_state().public_best or "unknown")
        judge_state = new_judge_state()

        def _judge_now() -> None:
            cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
            judge_cards_ready(zen, model, cards, our, state=judge_state)

        def _research_done() -> bool:
            if not cards_feasible(workspace, research_md):
                return False
            _judge_now()
            if judge_state["ready"] or judge_state["streak"] >= 2:
                append_daily_log(
                    "research judge ready="
                    f"{judge_state['ready']} streak={judge_state['streak']} "
                    f"{judge_state['last_reason']}",
                    self.root,
                )
                return True
            append_daily_log(
                f"research judge reject reason={judge_state['last_reason']}",
                self.root,
            )
            return False

        agent = ResearchAgent(
            zen,
            model,
            self._research_tools(result, deep_ran, judge_state=judge_state),
            self.settings.research_agent_config(),
            log=lambda msg: append_daily_log(msg, self.root),
            accept_done=_research_done,
            reject_msg="done rejected: judge says cards not ready; improve the cards",
            tracer=self._tracer,
            must_first=["harvest_cards"] if thin else [],
            must_first_args={"harvest_cards": {"reset": True}} if thin else None,
            stall_after=6,
            stall_nudge=(
                "Stall: you have read enough. Call write_card or harvest_cards "
                "now with your single best finding, then judge_cards, then done."
            ),
            stall_force=("done", {}),
        )
        pack = build_context_pack(self.root, view="research")
        out = agent.run(pack.as_prompt_block() or self.competition.slug)
        result.research_passes = max(1, out.turns)
        append_daily_log(
            f"research agent stop={out.stop_reason} turns={out.turns}",
            self.root,
        )
        if out.stop_reason in ("turn_cap", "time"):
            best = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
            append_daily_log(
                f"research budget exhausted stop={out.stop_reason} "
                f"best_so_far_cards={len(best)}",
                self.root,
            )
        harvested = any("wrote " in (o or "") and "card" in (o or "") for o in out.observations)
        if not cards_feasible(workspace, research_md) and not harvested:
            self._source_cards(result, reset=True)
            if self._tracer is not None:
                self._tracer.emit(
                    "tool", stage="research", tool="harvest_cards", source="safety_net"
                )
        if cards_feasible(workspace, research_md):
            append_daily_log("research cards feasible", self.root)
        else:
            append_daily_log("research cards still thin; continuing", self.root)
        deep = self.settings.deep_research_config()
        if deep.enabled and not deep_ran["n"]:
            self._deep_research(result)
        return state

    def _fleet_enabled(self) -> bool:
        cfg = self.settings.research_fleet_config()
        return bool(cfg.enabled or self.competition.fleet_enabled)

    def _fleet_roster(self) -> list[str]:
        comp = self.competition.fleet_agents
        roster = comp if comp else list(self.settings.research_fleet_config().agents)
        dropped = [a for a in roster if a not in AGENT_SPECS]
        if dropped:
            append_daily_log(f"research fleet unknown agents dropped: {dropped}", self.root)
        return [a for a in roster if a in AGENT_SPECS]

    def _fleet_research(self, state: AgentState, result: CycleResult) -> None:
        """Run one StageAgent loop per source in parallel; then converge cards."""
        workspace = self.root / self.competition.workspace_relative
        research_md = memory_dir(self.root) / "research.md"
        dest = memory_dir(self.root) / "research-deep"
        our = str(self._sa.load_state().public_best or "unknown")
        zen = self.router.client if self.router is not None else None
        model = self.competition.model_for("distill", self.settings)
        cfg = self.settings.research_fleet_config()
        roster = self._fleet_roster()
        agent_config = StageAgentConfig(
            max_minutes=cfg.max_minutes,
            max_tool_turns=cfg.max_tool_turns,
            max_tokens=cfg.max_tokens,
        )
        pack = build_context_pack(self.root, view="research")
        context = pack.as_prompt_block() or self.competition.slug
        thin = not cards_feasible(workspace, research_md)
        shared = self._source_tool_closures()
        search_fn = shared["search"]
        fetch_fn = shared["fetch_url"]
        kernel_list_fn = shared["list_kernels"]
        kernel_pull_fn = shared["pull_kernel"]
        harvested = {"n": 0}
        harvest_lock = threading.Lock()

        def harvest_cards(reset: bool | None = None, **_: Any) -> str:
            with harvest_lock:
                if harvested["n"] >= 1:
                    return "already harvested this run; call write_card or done"
                harvested["n"] += 1
            self._source_cards(result, reset=bool(reset))
            cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
            return f"wrote {len(cards)} cards"

        agents: list[tuple[str, StageAgent]] = []
        for name in roster:
            spec = AGENT_SPECS[name]
            wrote = {"n": 0}
            base_write = make_write_card(
                dest,
                spec.card_kind,
                agent=name,
                run_id=result.experiment_id or "unknown",
            )

            def _write(
                ref: str = "",
                markdown: str = "",
                *,
                _base_write=base_write,
                _wrote=wrote,
            ) -> str:
                path = _base_write(ref, markdown)
                _wrote["n"] += 1
                return path

            tools = make_fleet_tools(
                spec,
                search_fn=search_fn,
                fetch_fn=fetch_fn,
                write_fn=_write,
                kernel_list_fn=(
                    kernel_list_fn if "list_kernels" in spec.tools else None
                ),
                kernel_pull_fn=(
                    kernel_pull_fn if "pull_kernel" in spec.tools else None
                ),
                max_searches=2,
            )
            tools["harvest_cards"] = harvest_cards
            agents.append(
                (
                    name,
                    StageAgent(
                        clone_client_for_agent(zen),
                        model,
                        tools,
                        agent_config,
                        system=subagent_system(name, self.competition.slug, our),
                        log=lambda msg: append_daily_log(msg, self.root),
                        accept_done=lambda _wrote=wrote: _wrote["n"] > 0,
                        reject_msg="done rejected: write at least one card first",
                        must_first=["harvest_cards"] if thin else [],
                        must_first_args={"harvest_cards": {"reset": True}} if thin else None,
                        stall_after=6,
                        stall_nudge=(
                            "Stall: you have read enough. Call write_card now with "
                            "your single best finding (ref + markdown body). Call "
                            "done only after write_card succeeded."
                        ),
                        stall_force=None,
                        force_after_stall="write_card",
                        name="research",
                        agent_id=name,
                        tool_schemas=fleet_tool_schemas(spec),
                        tracer=self._tracer,
                    ),
                )
            )
        out = run_fleet(agents, log=lambda msg: append_daily_log(msg, self.root))
        result.research_passes = max(1, out.turns)
        research_verification = verify_research_fleet(out.executions, roster)
        result.research_verified = research_verification.ok
        result.research_verification_detail = research_verification.detail
        if self._tracer is not None:
            for execution in out.executions:
                self._tracer.emit(
                    "agent_execution",
                    stage="research",
                    agent=execution.agent,
                    stop_reason=execution.stop_reason,
                    turns=execution.turns,
                    tool_calls=execution.tool_calls,
                    source_reads=execution.source_reads,
                    writes=execution.writes,
                    rejected_writes=execution.rejected_writes,
                    errors=execution.errors,
                    verified=verify_research_fleet([execution], [execution.agent]).ok,
                )
        cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
        if cards:
            merge_digest(cards, research_md, our)
            write_methods_sidecar(cards, workspace)
            result.deep_ok = True
            result.deep_sources = max(result.deep_sources, len(cards))
        if self._tracer is not None:
            self._tracer.emit(
                "tool",
                stage="research",
                tool="fleet",
                agents=len(agents),
                required_agents=roster,
                turns=out.turns,
                verified=result.research_verified,
                verification_detail=research_verification.detail,
                errors=out.errors[:3],
            )
        append_daily_log(
            f"research fleet agents={len(agents)} turns={out.turns} "
            f"cards={len(cards)} verified={result.research_verified} "
            f"verification={research_verification.detail} errors={out.errors or 'none'}",
            self.root,
        )
        if not cards_feasible(workspace, research_md) and not harvested["n"]:
            self._source_cards(result, reset=True)
        if cards_feasible(workspace, research_md):
            append_daily_log("research cards feasible", self.root)
            ready, reason = judge_cards_ready(zen, model, cards, our)
            append_daily_log(
                f"research judge post-fleet ready={ready} {reason}",
                self.root,
            )
            if not ready:
                self._fleet_polish(result, reason)
        else:
            append_daily_log("research cards still thin; continuing", self.root)

    def _fleet_polish(self, result: CycleResult, reason: str) -> None:
        """One bounded StageAgent pass: judge, rewrite the weakest card, re-judge."""
        workspace = self.root / self.competition.workspace_relative
        research_md = memory_dir(self.root) / "research.md"
        dest = memory_dir(self.root) / "research-deep"
        our = str(self._sa.load_state().public_best or "unknown")
        zen = self.router.client if self.router is not None else None
        model = self.competition.model_for("distill", self.settings)
        cfg = self.settings.research_fleet_config()
        judge_state = new_judge_state()
        wrote = {"n": 0}
        reads = {"n": 0}
        base_write = make_write_card(dest, "polish")

        def read_cards(**_: Any) -> str:
            cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
            parts = []
            for p in cards[-6:]:
                parts.append(f"### {p.name}\n{p.read_text(encoding='utf-8')[:2000]}")
            text = "\n\n".join(parts) or "no cards"
            if reads["n"] > 0:
                text += (
                    "\n\nYou already read the cards. Call write_card now with an "
                    "improved card; do not read again."
                )
            reads["n"] += 1
            return text

        def write_card(ref: str = "", markdown: str = "", **_: Any) -> str:
            path = base_write(ref, markdown)
            wrote["n"] += 1
            return path

        def judge_cards(**_: Any) -> str:
            cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
            ready, r = judge_cards_ready(zen, model, cards, our, state=judge_state)
            feasible = cards_feasible(workspace, research_md)
            return f"ready={ready} feasible={feasible} {r}"

        agent = StageAgent(
            clone_client_for_agent(zen),
            model,
            {
                "read_cards": read_cards,
                "write_card": write_card,
                "judge_cards": judge_cards,
            },
            StageAgentConfig(
                max_minutes=cfg.max_minutes,
                max_tool_turns=min(cfg.max_tool_turns, 12),
                max_tokens=cfg.max_tokens,
            ),
            system=(
                "You polish one Kaggle contest's research cards. "
                "Call one tool per turn. "
                f"The research judge said: {reason}. "
                "Call read_cards, then write_card to improve the weakest card, "
                "then judge_cards. Call done only after write_card succeeded "
                "or the judge approves. "
                "Card body format (write_card rejects bodies missing these):\n"
                "- copyable next step: <one implementable change>\n"
                "- do not copy: <anti-patterns to avoid>\n"
            ),
            log=lambda msg: append_daily_log(msg, self.root),
            accept_done=lambda: judge_state["ready"] is True
            or judge_state["streak"] >= 2
            or wrote["n"] > 0,
            reject_msg="done rejected: judge says not ready; rewrite a weaker card",
            must_first=["read_cards"],
            stall_after=4,
            stall_nudge=(
                "Stall: call write_card with an improved card addressing the "
                "judge, then judge_cards."
            ),
            stall_force=("done", {}),
            name="research",
            tracer=self._tracer,
        )
        pack = build_context_pack(self.root, view="research")
        out = agent.run(pack.as_prompt_block() or self.competition.slug)
        result.research_passes = max(result.research_passes, out.turns)
        append_daily_log(
            f"research polish stop={out.stop_reason} turns={out.turns} "
            f"wrote={wrote['n']}",
            self.root,
        )
        cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
        if cards:
            merge_digest(cards, research_md, our)
            write_methods_sidecar(cards, workspace)
        ready, r = judge_cards_ready(zen, model, cards, our)
        append_daily_log(
            f"research judge post-polish ready={ready} {r}",
            self.root,
        )

    def _source_tool_closures(self) -> dict[str, Callable[..., str]]:
        """Source closures shared by the sequential research loop and the fleet."""
        cache = (self.root / self.competition.workspace_relative) / "research-cache"

        def list_kernels(query: str = "", limit: int = 6, **_: Any) -> str:
            if self._kaggle is None:
                return "no kaggle client"
            rows = self._kaggle.kernels(self.competition.slug, top=int(limit) + 2)
            refs = [r.ref for r in rows if r.ref][: int(limit)]
            if query:
                refs = [r for r in refs if query.lower() in r.lower()]
            return "\n".join(refs) or "none"

        def pull_kernel(ref: str = "", **_: Any) -> str:
            if self._kaggle is None or not ref:
                return "missing ref"
            src = KaggleSource(self._kaggle, self.competition.slug, cache)
            from kaggle_agent.research.deep import SourceHit

            hit = SourceHit(url=ref, title=ref, kind="kaggle")
            with _PULL_LOCK:
                return src.content(hit)[:12000]

        def fetch_url(url: str = "", **_: Any) -> str:
            if url and not url.lower().startswith(("http://", "https://")):
                return "refuse: only http(s) urls"
            if self._browser_fetch is not None:
                return str(self._browser_fetch(url, 12000))[:12000]
            from kaggle_agent.research.browser import fetch_via_http

            return fetch_via_http(url, 12000)[:12000]

        def search(query: str = "", kind: str = "web", limit: int = 5, **_: Any) -> str:
            sources = {
                "kaggle": (
                    KaggleSource(self._kaggle, self.competition.slug, cache)
                    if self._kaggle is not None
                    else None
                ),
                "arxiv": ArxivSource(),
                "github": GithubSource(),
                "web": WebSource(serp=default_serp(self.settings.browser_prefer_harness)),
                "discussion": (
                    DiscussionSource(self._kaggle, self.competition.slug)
                    if self._kaggle is not None
                    else None
                ),
                "dataset": (
                    DatasetSource(self._kaggle, self.competition.slug)
                    if self._kaggle is not None
                    else None
                ),
            }
            src = sources.get(str(kind), sources["web"])
            if src is None:
                return "no source"
            hits = src.search(str(query), int(limit))
            return "\n".join(f"{h.kind}\t{h.url}\t{h.title}" for h in hits) or "none"

        return {
            "list_kernels": list_kernels,
            "pull_kernel": pull_kernel,
            "fetch_url": fetch_url,
            "search": search,
        }

    def _research_tools(
        self,
        result: CycleResult,
        deep_ran: dict[str, int] | None = None,
        judge_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workspace = self.root / self.competition.workspace_relative
        dest = memory_dir(self.root) / "research-deep"
        our = str(self._sa.load_state().public_best or "unknown")
        zen = self.router.client if self.router is not None else None
        model = self.competition.model_for("distill", self.settings)
        shared = self._source_tool_closures()
        list_kernels = shared["list_kernels"]
        pull_kernel = shared["pull_kernel"]
        fetch_url = shared["fetch_url"]
        search = shared["search"]

        def write_card(ref: str = "", markdown: str = "", **_: Any) -> str:
            dest.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^a-z0-9]+", "-", (ref or "src").lower()).strip("-")[:60]
            path = dest / f"source-{slug or 'src'}.md"
            path.write_text(markdown or f"# {ref}\n- ref: {ref}\n", encoding="utf-8")
            cards = sorted(dest.glob("source-*.md"))
            merge_digest(cards, memory_dir(self.root) / "research.md", our)
            write_methods_sidecar(cards, workspace)
            result.deep_ok = True
            result.deep_sources = max(result.deep_sources, len(cards))
            return str(path)

        def judge_cards(**_: Any) -> str:
            cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
            ready, reason = judge_cards_ready(zen, model, cards, our, state=judge_state)
            feasible = cards_feasible(workspace, memory_dir(self.root) / "research.md")
            return f"ready={ready} feasible={feasible} {reason}"

        harvested = {"n": 0}

        def harvest_cards(reset: bool | None = None, **_: Any) -> str:
            if harvested["n"] >= 1:
                return "already harvested this run; call judge_cards or done"
            research_md = memory_dir(self.root) / "research.md"
            if reset is None:
                reset = not cards_feasible(workspace, research_md)
            self._source_cards(result, reset=bool(reset))
            harvested["n"] += 1
            cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
            return f"wrote {len(cards)} cards"

        def deep_research(**_: Any) -> str:
            if deep_ran is not None:
                deep_ran["n"] += 1
            self._deep_research(result)
            return f"deep_ok={result.deep_ok} sources={result.deep_sources}"

        return {
            "list_kernels": list_kernels,
            "pull_kernel": pull_kernel,
            "fetch_url": fetch_url,
            "search": search,
            "write_card": write_card,
            "judge_cards": judge_cards,
            "harvest_cards": harvest_cards,
            "deep_research": deep_research,
        }

    def _kaggle_snapshot(self, state: AgentState, result: CycleResult) -> None:
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

    def _cards_judged_ready(self) -> bool:
        dest = memory_dir(self.root) / "research-deep"
        cards = sorted(dest.glob("source-*.md")) if dest.is_dir() else []
        zen = self.router.client if self.router is not None else None
        model = self.competition.model_for("distill", self.settings)
        our = str(self._sa.load_state().public_best or "unknown")
        ready, reason = judge_cards_ready(zen, model, cards, our)
        append_daily_log(f"research judge ready={ready} {reason}", self.root)
        return ready

    def _source_cards(self, result: CycleResult, *, reset: bool = True) -> None:
        """One Zen worker per kernel, discussion, and paper. New cards each pass."""
        if self._kaggle is None:
            return
        try:
            our = str(self._sa.load_state().public_best or "unknown")
            cache = self.root / self.competition.workspace_relative / "research-cache"
            zen = self.router.client if self.router is not None else None
            model = self.competition.model_for("distill", self.settings)
            cards = run_source_card_research(
                client=self._kaggle,
                competition=self.competition.slug,
                cache_dir=cache,
                root=self.root,
                our_score=our,
                zen=zen,
                model=model,
                reset=reset,
                log=lambda msg: append_daily_log(msg, self.root),
            )
            for path in cards:
                lines = path.read_text(encoding="utf-8").splitlines()
                if "- agent:" not in lines[:5]:
                    insert_at = 1 if lines and lines[0].startswith("#") else 0
                    lines[insert_at:insert_at] = [
                        "- agent: fallback",
                        f"- run_id: {result.experiment_id or 'unknown'}",
                    ]
                    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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

    def _deep_research(
        self, result: CycleResult, *, max_queries: int | None = None
    ) -> None:
        """Deep-research stage: recursive search over web/papers/notebooks/repos."""
        deep = self.settings.deep_research_config()
        if max_queries is not None:
            deep = replace(deep, max_queries=max_queries)
        if not deep.enabled:
            return
        try:
            if self._kaggle is None:
                self._kaggle = KaggleClient().connect()
            zen = self.router.client if self.router is not None else None
            if zen is None:
                append_daily_log("deep research skipped: no DEEPSEEK_API_KEY", self.root)
                return
            model = self.competition.model_for("distill", self.settings)
            cache = self.root / self.competition.workspace_relative / "research-cache"
            prompt = self._deep_prompt()
            web_fetch = self._browser_fetch or default_fetch(
                self.settings.browser_prefer_harness
            )
            web_serp = default_serp(self.settings.browser_prefer_harness)
            terms = tuple(
                dict.fromkeys(
                    [self.competition.slug] + self.competition.slug.split("-")[:2]
                )
            )
            researcher = DeepResearcher(
                zen,
                model,
                deep,
                sources=[
                    KaggleSource(self._kaggle, self.competition.slug, cache),
                    ArxivSource(),
                    GithubSource(),
                    WebSource(fetch=web_fetch, serp=web_serp),
                ],
                root=self.root,
                log=lambda msg: append_daily_log(msg, self.root),
                relevance_terms=terms,
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
        if not check.ok:
            result.code_ok = False
            result.errors.append(f"code: missing {check.missing}")
            append_daily_log(f"code missing={check.missing}", self.root)
            if self._tracer is not None:
                self._tracer.emit(
                    "agent_verification",
                    stage="code",
                    agent="code",
                    verified=False,
                    detail="pipeline missing",
                )
            return state
        zen = self.router.client if self.router is not None else None
        model = self.competition.model_for("code", self.settings)
        agent, code_state = make_code_agent(
            zen,
            model,
            self.root,
            workspace,
            self.settings.code_agent_config(),
            plan_text=result.plan_text or "",
            log=lambda msg: append_daily_log(msg, self.root),
            tracer=self._tracer,
        )
        pack = build_context_pack(
            self.root,
            view="code",
            workspace=workspace,
            plan_text=result.plan_text or "",
        )
        out = agent.run(
            f"Competition: {self.competition.slug}\n\n{pack.as_prompt_block()}"
        )
        result.code_agent = out.agent
        if self._tracer is not None:
            self._tracer.emit(
                "agent_execution",
                stage="code",
                agent=out.agent,
                stop_reason=out.stop_reason,
                turns=out.turns,
                tool_calls=out.tool_calls,
                writes=out.writes,
                rejected_writes=out.rejected_writes,
                errors=out.errors,
                verified=bool(out.writes),
            )
        result.wrote_custom_infer = bool(code_state.get("wrote_custom_infer"))
        result.wrote_methods = bool(code_state.get("wrote_methods"))
        result.wrote_recipe = bool(code_state.get("wrote_recipe"))
        if self._tracer is not None:
            self._tracer.emit(
                "tool",
                stage="code",
                tool="code_hook",
                source="written" if result.wrote_custom_infer else "identity",
                recipe="written" if result.wrote_recipe else "static",
            )
        append_daily_log(
            f"code agent stop={out.stop_reason} turns={out.turns} "
            f"wrote_methods={result.wrote_methods} wrote_recipe={result.wrote_recipe} "
            f"wrote_custom_infer={result.wrote_custom_infer}",
            self.root,
        )
        if not (result.wrote_recipe or result.wrote_custom_infer):
            result.code_ok = False
            result.errors.append("code: no recipe change was written")
            append_daily_log("code rejected: no recipe change was written", self.root)
            if self._tracer is not None:
                self._tracer.emit(
                    "agent_verification",
                    stage="code",
                    agent=out.agent,
                    verified=False,
                    detail="no implementation artifact",
                )
            return state
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
        except Exception as exc:  # noqa: BLE001
            result.code_ok = False
            result.errors.append(f"code: recipe failed: {exc}")
            append_daily_log(f"code recipe failed: {exc}", self.root)
        if self._tracer is not None:
            self._tracer.emit(
                "agent_verification",
                stage="code",
                agent=out.agent,
                verified=bool(result.code_ok),
                detail="recipe validation",
            )
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
        if result.wrote_recipe or result.wrote_custom_infer:
            result.code_verified = verify_code_stage(
                wrote_recipe=result.wrote_recipe,
                wrote_custom_infer=result.wrote_custom_infer,
                artifact_ok=bool(result.code_ok),
                smoke_ok=bool(result.smoke_ok),
            ).ok
            if not result.code_verified:
                result.code_ok = False
                result.errors.append("code verification: artifact or smoke check failed")
            if self._tracer is not None:
                self._tracer.emit(
                    "agent_verification",
                    stage="code",
                    agent=result.code_agent or "code",
                    verified=result.code_verified,
                    smoke_ok=result.smoke_ok,
                )
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
        existing = self._sa.load_kernel_job()

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
                if should_push and self._kaggle is None:
                    self._kaggle = KaggleClient().connect()
                package = write_kernel_package(
                    self.competition,
                    root=self.root,
                    username=self._kernel_username(),
                    exp_id=exp_id,
                    enable_gpu=self.settings.kernel_enable_gpu,
                    machine_shape=self.settings.kernel_machine_shape,
                    enable_internet=self.settings.kernel_enable_internet,
                    plan_text=result.plan_text or "",
                )
                result.kernel_path = str(package.folder)
                result.kernel_ref = package.kernel_ref
                out_dir = package.folder / "output"
                if should_push and package_matches_existing(package, existing):
                    # Same notebook + metadata as a previous kernel: CODE/PLAN
                    # produced no real change. Stop instead of silently
                    # resubmitting the same kernel as if it were a new one.
                    result.kernel_duplicate = True
                    run = KernelRunResult(
                        ok=False,
                        package=package,
                        resumed=False,
                        kernel_ref=existing.kernel_ref,
                        message=f"identical kernel: no change since {existing.kernel_ref}",
                        status=existing.status,
                        errors=[
                            f"kernel is identical to previous experiment {existing.kernel_ref}"
                        ],
                    )
                else:
                    run = run_kernel_phase(
                        self._kaggle if should_push else None,
                        package,
                        push=should_push,
                        pull_output_dir=out_dir if should_push else None,
                        root=self.root,
                        competition=self.competition.slug,
                        exp_id=exp_id,
                        poll_seconds=self.settings.kernel_poll_seconds,
                        poll_attempts=self.settings.kernel_poll_attempts,
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
        if not result.wrote_custom_infer:
            if result.smoke_path:
                candidates.append(Path(result.smoke_path))
            exp_id = result.experiment_id
            if exp_id:
                same = (
                    self.root
                    / self.competition.workspace_relative
                    / "submissions"
                    / f"{exp_id}_smoke.csv"
                )
                candidates.append(same)

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
            kernel_output = (
                result.kernel_path is not None
                and path == Path(result.kernel_path) / "output" / "submission.csv"
            )
            if kernel_output:
                out_hash = submission_output_hash(path, self.competition.id_column)
                prior_exp = seen_output(
                    self.root, out_hash, exp_id=result.experiment_id or ""
                )
                if prior_exp and not os.environ.get("KAGGLE_AGENT_ALLOW_DUPLICATE_OUTPUT"):
                    result.output_duplicate = True
                    result.validate_ok = False
                    result.errors.append(
                        f"validate: predictions identical to previous experiment {prior_exp}"
                    )
                    append_daily_log(
                        f"validate rejected: output identical to {prior_exp}", self.root
                    )
                else:
                    record_output(self.root, result.experiment_id or "unknown", out_hash)
        else:
            result.errors.extend(f"validate:{e}" for e in check.errors[:5])
            append_daily_log(f"validate failed: {check.errors[:3]}", self.root)

        kernel_judge = new_judge_state()
        kernel_csv = (
            Path(result.kernel_path) / "output" / "submission.csv"
            if result.kernel_path
            else None
        )
        if kernel_csv is not None and kernel_csv.is_file():
            job = self._sa.load_kernel_job()

            def judge_log(msg: str) -> None:
                append_daily_log(msg, self.root)

            ready, _reason = judge_kernel(job.status, check, state=kernel_judge, log=judge_log)
            result.kernel_judge_ok = ready
            if ready and self.settings.judge_train:
                zen = self.router.client if self.router is not None else None
                if zen is not None:
                    ready, _reason = judge_train_llm(
                        zen,
                        self.competition.model_for("train", self.settings),
                        job.status,
                        path,
                        self.competition.labels,
                        str(self._sa.load_state().public_best or "unknown"),
                        state=kernel_judge,
                        log=judge_log,
                    )
                    result.kernel_judge_ok = ready
            if kernel_judge.get("last_reason"):
                patch_experiment(
                    result.experiment_id or "unknown",
                    judge=f"kernel {kernel_judge['ready']}: {kernel_judge['last_reason']}",
                    root=self.root,
                )
        elif self.settings.judge_train:
            append_daily_log("judge train skipped: no kernel output", self.root)
        elif kernel_csv is None:
            append_daily_log("judge kernel skipped: no kernel path", self.root)
        else:
            job = self._sa.load_kernel_job()
            append_daily_log(
                f"judge kernel skipped: no kernel output (status={job.status})",
                self.root,
            )
        return state

    def _plan(self, state: AgentState, dry: bool, result: CycleResult) -> None:
        hypothesis, approach, notes = DEFAULT_HYPOTHESIS, "baseline", "plan agent"
        workspace = self.root / self.competition.workspace_relative
        pack = build_context_pack(self.root, view="plan", workspace=workspace)
        zen = self.router.client if self.router is not None else None
        model = self.competition.model_for("plan", self.settings)

        def on_plan(h: str, a: str, steps: str) -> None:
            nonlocal hypothesis, approach
            hypothesis, approach = h, a
            result.plan_text = write_plan_text(h, a, steps)

        judge_state = new_judge_state()

        def plan_judge(h: str, a: str, steps: str) -> tuple[bool, str]:
            try:
                methods = load_methods(workspace)
            except Exception:  # noqa: BLE001
                methods = {}
            return judge_plan(
                zen,
                model,
                write_plan_text(h, a, steps),
                methods,
                str(state.public_best or "unknown"),
                state=judge_state,
                log=lambda msg: append_daily_log(msg, self.root),
            )

        agent, _state = make_plan_agent(
            zen,
            model,
            self.root,
            self.settings.plan_agent_config(),
            workspace=workspace,
            log=lambda msg: append_daily_log(msg, self.root),
            on_plan=on_plan,
            judge=plan_judge,
            tracer=self._tracer,
        )
        out = agent.run(
            f"Competition: {self.competition.slug}\n\n{pack.as_prompt_block()}"
        )
        append_daily_log(f"plan agent stop={out.stop_reason} turns={out.turns}", self.root)
        result.plan_verified = verify_plan_stage(
            wrote=bool(_state.get("wrote")),
            judge_ready=bool(judge_state.get("ready")),
        ).ok
        if self._tracer is not None:
            self._tracer.emit(
                "agent_execution",
                stage="plan",
                agent=out.agent,
                stop_reason=out.stop_reason,
                turns=out.turns,
                tool_calls=out.tool_calls,
                writes=out.writes,
                rejected_writes=out.rejected_writes,
                errors=out.errors,
                verified=result.plan_verified,
            )
            self._tracer.emit(
                "agent_verification",
                stage="plan",
                agent=out.agent,
                verified=result.plan_verified,
            )
        if result.plan_text is None:
            methods = load_methods(workspace)
            steps = [
                s
                for s in (methods.get("implement_steps") or [])
                if s and "dry-run default" not in str(s).lower()
            ]
            if steps:
                hypothesis, approach = str(steps[0])[:240], "recipe"
                result.plan_text = write_plan_text(hypothesis, approach, "; ".join(steps[:3]))
            else:
                result.plan_text = write_plan_text(hypothesis, approach, "")
        notes = f"plan agent {out.stop_reason}"
        if zen is None or not hasattr(zen, "chat"):
            notes = "DEEPSEEK_API_KEY missing; offline plan from cards"

        write_experiment(
            result.experiment_id or "unknown",
            hypothesis=hypothesis,
            approach=approach,
            notes=notes + ("; dry_run" if dry else ""),
            root=self.root,
        )
        if judge_state.get("last_reason"):
            patch_experiment(
                result.experiment_id or "unknown",
                judge=f"plan {judge_state['ready']}: {judge_state['last_reason']}",
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

        if self._assume_approved:
            request_approval(
                exp_id=exp_id,
                csv_path=csv_path,
                competition=self.competition.slug,
                message="assume_approved",
                root=self.root,
                kernel_path=result.kernel_path or "none",
                kernel_ref=result.kernel_ref or "none",
            )
            set_decision("latest", approved=True, root=self.root)
            result.approve_ok = True
            result.waiting_approve = False
            state.pending_approve = exp_id
            append_daily_log(f"approve assumed exp={exp_id}", self.root)
            return state

        # Live: keep prior /yes so a second /run can submit without re-asking
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

        pending = self._sa.load_pending()
        if not dry and self.settings.require_telegram_approve and not self._assume_approved:
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
                    "Send /yes now, then /run again.\n"
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
            self._sa.save_pending(pending)

        if (
            not dry
            and self.settings.block_submit
        ):
            report = memory_dir(self.root) / "daily" / "eval_report.json"
            try:
                import json

                ev = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
            except Exception:  # noqa: BLE001
                ev = {}
            if ev.get("passed") is False:
                result.submit_ok = False
                result.submit_message = "eval gate closed"
                heal = load_heal(self.root)
                heal.note = "eval gate closed"
                save_heal(heal, self.root)
                append_daily_log("submit skipped: eval gate closed", self.root)
                return state

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
                            poll_attempts=(
                                self.settings.kernel_poll_attempts if nb else 10
                            ),
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
            if not subs:
                append_daily_log("feedback: no submissions yet", self.root)
                return state
            latest = subs[0]
            append_daily_log(
                f"feedback status={latest.status} score={latest.public_score}",
                self.root,
            )
            scored = (
                latest
                if parse_loop_score(latest.public_score) is not None
                else self._wait_for_feedback_score()
            )
            if scored is None:
                append_daily_log("feedback: no scored submission yet", self.root)
                return state
            result.feedback_score = scored.public_score
            pending = self._sa.load_pending()
            exp_id = (
                pending.exp_id
                if pending.exp_id not in {"", "none"}
                else result.experiment_id
            )
            if exp_id and exp_id != "none":
                patch_experiment(
                    exp_id,
                    root=self.root,
                    public_score=scored.public_score,
                    submission=scored.status or "submitted",
                    kernel=result.kernel_ref or pending.kernel_ref,
                )
            self._apply_best_score(state, scored.public_score)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"feedback: {exc}")
            append_daily_log(f"feedback failed: {exc}", self.root)
        return state

    def _wait_for_feedback_score(self) -> SubmissionRow | None:
        """Poll the LB until the newest submission has a numeric score.

        Bounded by settings.feedback_wait_minutes; 0 disables the wait.
        """
        wait_minutes = self.settings.feedback_wait_minutes
        if wait_minutes <= 0:
            return None
        poll = self.settings.feedback_poll_seconds
        deadline = time.monotonic() + wait_minutes * 60
        while time.monotonic() < deadline:
            time.sleep(poll)
            subs = self._kaggle.submissions(self.competition.slug, top=3)
            latest = subs[0]
            scored = (
                latest
                if parse_loop_score(latest.public_score) is not None
                else None
            )
            if scored is not None:
                append_daily_log(
                    f"feedback waited: score={scored.public_score}", self.root
                )
                return scored
            append_daily_log("feedback: still pending score, polling…", self.root)
        append_daily_log(f"feedback: no score within {wait_minutes}m", self.root)
        return None

    def _catch_up_scores(self, state: AgentState) -> None:
        """Ingest scores that landed after a previous cycle exited.

        Patches experiment files, then advances heal/loop state and the
        public best, so a late score still drives the self-heal ladder.
        """
        if self._kaggle is None:
            self._kaggle = KaggleClient().connect()
        subs = self._kaggle.submissions(self.competition.slug, top=20)
        ingested: list[SubmissionRow] = []
        for row in subs:
            if parse_loop_score(row.public_score) is None:
                continue
            exp_id = exp_id_from_description(row.description)
            if not exp_id:
                continue
            exp_file = memory_dir(self.root) / "experiments" / f"{exp_id}.md"
            if not exp_file.is_file() or already_recorded(exp_file, row.public_score):
                continue
            patch_experiment(
                exp_id,
                root=self.root,
                public_score=row.public_score,
                submission=row.status or "submitted",
            )
            ingested.append(row)
            append_daily_log(
                f"score catch-up exp={exp_id} score={row.public_score}", self.root
            )
        if not ingested:
            return
        self._advance_heal_and_loop(ingested[0], state)

    def _advance_heal_and_loop(self, row: SubmissionRow, state: AgentState) -> None:
        score = row.public_score
        heal = load_heal(self.root)
        if parse_loop_score(heal.last_score) != parse_loop_score(score):
            heal = decide_next(
                heal,
                public_score=score,
                metric_direction=self.competition.metric_direction,
                max_tune_attempts=self.settings.max_tune_attempts,
                max_no_improve_days=self.settings.max_no_improve_days,
                cycle_ok=True,
            )
            save_heal(heal, self.root)
            append_daily_log(
                f"heal catch-up score={score} decision_next={heal.decision_next}",
                self.root,
            )
        try:
            n_used = int(load_loop(self.root).last_n)
        except (TypeError, ValueError):
            n_used = 1
        update_loop_from_score(
            self.root,
            score,
            n_used=n_used,
            n_min=self.settings.loop_n_min,
            n_max=self.settings.loop_n_max,
            typical_gain=self.settings.loop_typical_gain,
            default_n=self.settings.loop_default_n,
            direction=self.competition.metric_direction,
        )
        self._apply_best_score(state, score)

    def _apply_best_score(self, state: AgentState, score: str) -> None:
        """Raise the public best and persist it when a score improves it."""
        if score_is_better(score, state.public_best, self.competition.metric_direction):
            state.public_best = score
            patch_memory_public_score(str(score), self.root)

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
        from kaggle_agent.heal.pins import apply_pin_heal, is_pin_error, should_wait_approve

        pin_errs = [e for e in result.errors if is_pin_error(e)]
        if pin_errs and result.kernel_path:
            workspace = self.root / self.competition.workspace_relative
            healed = apply_pin_heal(workspace, Path(result.kernel_path))
            append_daily_log(
                f"heal pin strip changed={healed.get('changed')} "
                f"models={healed.get('model_sources')} datasets={healed.get('dataset_sources')}",
                self.root,
            )
        if should_wait_approve(
            validate_ok=result.validate_ok,
            submit_ok=result.submit_ok,
            dry_run=result.dry_run,
            assume_approved=self._assume_approved,
            errors=result.errors,
        ):
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
                _flag("CUSTOM_INFER hook", result.wrote_custom_infer),
                _flag("Kernel recipe", result.wrote_recipe),
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
                "2) Send /run so it can submit to Kaggle\n"
            )
        elif hard:
            headline = f"Cycle finished with problems — {mode}"
            next_step = "Next: check Errors below, then /status or /run again.\n"
        else:
            headline = f"Cycle finished — {mode}"
            next_step = "Next: /status · /run · /help\n"

        slices = result.train_slices
        slice_line = (
            f"Train slices: {slices}\n" if slices else ""
        )
        passes = result.research_passes
        pass_line = f"Research passes: {passes}\n" if passes else ""

        text = (
            f"{headline}\n\n"
            f"Experiment: {result.experiment_id}\n"
            f"Competition: {self.competition.slug}\n"
            f"{slice_line}{pass_line}\n"
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
    assume_approved: bool = False,
    kaggle: KaggleClient | None = None,
    browser_fetch: FetchFn | None = None,
    telegram: SupportsTelegram | None = None,
    browser_submit: BrowserSubmitFn | None = None,
    mcp_submit_fn: Any | None = None,
    skip_phases: frozenset[str] | None = None,
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
        skip_phases=skip_phases,
    ).run_cycle(dry_run=dry_run, assume_approved=assume_approved)
