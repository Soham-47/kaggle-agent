from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace as _copy_min
from kaggle_agent.config import DEFAULT_PHASES
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.orchestrator import run_daily
from kaggle_agent.state_md import AgentState, load_state, save_state


def _fake_kaggle() -> KaggleClient:
    return KaggleClient(api=FakeKaggleApi()).connect()


def _fake_browser(url: str, max_chars: int = 12000) -> str:
    return (
        "Competition overview: detect twelve knee abnormalities from multimodal MRI. "
        "Evaluation uses macro-averaged ROC AUC. "
        "Discussion tip: study-level 2D CNN baselines are strong starters. "
    ) * 2


def test_dry_run_cycle(tmp_path: Path):
    from kaggle_agent.notify.telegram import FakeTelegram

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    tg = FakeTelegram()
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=True,
        kaggle=_fake_kaggle(),
        browser_fetch=_fake_browser,
        telegram=tg,
    )
    assert not result.skipped
    assert result.kaggle_ok is True
    assert result.browser_ok is True
    assert result.code_ok is True
    assert result.smoke_ok is True
    assert result.smoke_path
    assert Path(result.smoke_path).is_file()
    assert result.kernel_ok is True
    assert result.kernel_path
    assert (Path(result.kernel_path) / "kernel-metadata.json").is_file()
    assert result.validate_ok is True
    assert result.candidate_csv
    assert result.approve_ok is True
    assert result.submit_ok is True  # dry submit
    assert result.phases_run[:4] == ["LOCK", "RESEARCH", "PLAN", "CODE"]
    st = load_state(root)
    assert st.phase == "IDLE"
    assert st.lock_held is False
    research = (root / "memory" / "research.md").read_text(encoding="utf-8")
    assert "Browser (read-only)" in research
    assert "allowed_now" in research or "Alpha" in research


def test_skip_when_paused(tmp_path: Path):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    save_state(AgentState(paused=True), root)
    r = run_daily(
        "rsna_knee",
        root=root,
        dry_run=True,
        kaggle=_fake_kaggle(),
        browser_fetch=_fake_browser,
    )
    assert r.skipped and r.skip_reason == "paused"


def test_dropped_submit_phase_is_skipped(tmp_path: Path):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    path = root / "config" / "settings.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("    - SUBMIT\n", ""), encoding="utf-8")
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=True,
        kaggle=_fake_kaggle(),
        browser_fetch=_fake_browser,
    )
    assert "SUBMIT" not in result.phases_run
    assert "RESEARCH" in result.phases_run
    assert result.phases_run.index("RESEARCH") < result.phases_run.index("PLAN")
    assert "VALIDATE_SUB" in result.phases_run
    assert "REPORT" in result.phases_run


def test_pending_kernel_stops_before_validation_submission_and_heal(monkeypatch, tmp_path: Path):
    from kaggle_agent.train.kernel_job import KernelJob, load_kernel_job, save_kernel_job

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    folder = root / "competitions" / "rsna_knee" / "notebooks" / "active"
    folder.mkdir(parents=True)
    save_kernel_job(
        KernelJob(
            kernel_ref="tester/active",
            folder=str(folder),
            status="RUNNING",
            competition="rsna-knee-abnormality-detection",
            exp_id="active",
        ),
        root,
    )
    api = FakeKaggleApi(status_queue=["RUNNING"])
    monkeypatch.setattr("kaggle_agent.train.kernel_runner.time.sleep", lambda _seconds: None)

    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        kaggle=KaggleClient(api=api).connect(),
        skip_phases=frozenset({"RESEARCH", "PLAN", "CODE", "LOCAL_SMOKE"}),
    )

    assert result.kernel_pending is True
    assert result.kernel_ok is None
    assert result.hard_errors == []
    assert "VALIDATE_SUB" not in result.phases_run
    assert "TELEGRAM_APPROVE" not in result.phases_run
    assert "SUBMIT" not in result.phases_run
    assert "HEAL" not in result.phases_run
    assert load_kernel_job(root).kernel_ref == "tester/active"


def _judge_orch(root: Path, zen=None):  # noqa: ANN001
    from kaggle_agent.config import load_competition, load_settings
    from kaggle_agent.orchestrator import Orchestrator

    settings = load_settings(root)
    comp = load_competition("rsna_knee", root)

    class _Router:
        def __init__(self, client) -> None:  # noqa: ANN001
            self.client = client

    return Orchestrator(settings, comp, root=root, router=_Router(zen))


def test_supervisor_mode_returns_failures_to_parent_without_inline_debug(tmp_path: Path):
    from helpers import copy_min_workspace
    from kaggle_agent.config import load_competition, load_settings
    from kaggle_agent.orchestrator import Orchestrator

    root = tmp_path / "kaggle-agent"
    copy_min_workspace(root, Path(__file__).resolve().parents[1])
    config_path = root / "config" / "settings.yaml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\nsupervisor:\n  enabled: true\n  mode: observe\n", encoding="utf-8")
    settings = load_settings(root)
    comp = load_competition("rsna_knee", root)
    sentinel = lambda *_args: "should not run"
    orchestrator = Orchestrator(settings, comp, root=root, debug_runner=sentinel)
    assert orchestrator._debug_runner is None


def test_code_missing_artifact_retries_once_with_fresh_agent(monkeypatch, tmp_path: Path):
    from kaggle_agent.agents.loop import StageAgentResult
    from kaggle_agent.orchestrator import CycleResult

    root = tmp_path / "kaggle-agent"
    _copy_min(root, Path(__file__).resolve().parents[1])
    orchestrator = _judge_orch(root, zen=object())
    sessions: list[object] = []

    class _StalledAgent:
        def run(self, _context: str) -> StageAgentResult:
            return StageAgentResult(stop_reason="stalled", turns=3, agent="code")

    def make_agent(*_args, **_kwargs):  # noqa: ANN001
        session = object()
        sessions.append(session)
        return _StalledAgent(), {}

    monkeypatch.setattr("kaggle_agent.orchestrator.make_code_agent", make_agent)
    result = CycleResult(
        competition="rsna_knee",
        dry_run=False,
        plan_text="steps: rank average predictions",
    )

    orchestrator._code(AgentState(), result)

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert result.code_ok is False
    assert result.code_outcome == "NO_IMPLEMENTABLE_PLAN"
    assert result.errors == ["code: no recipe change was written"]


def _write_kernel_output(root: Path, comp, exp_id: str) -> Path:  # noqa: ANN001
    kernel_dir = root / "competitions" / "rsna_knee" / "notebooks" / exp_id
    (kernel_dir / "output").mkdir(parents=True)
    header = ",".join([comp.id_column, *comp.labels])
    rows = [
        f"s{i}," + ",".join([str(0.5 + (i % 2) * 0.1)] * len(comp.labels))
        for i in range(1000)
    ]
    (kernel_dir / "output" / "submission.csv").write_text(
        header + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    return kernel_dir


def _write_exp(root: Path, exp_id: str) -> None:
    (root / "memory" / "experiments" / f"{exp_id}.md").write_text(
        f"# {exp_id}\n\n- hypothesis: h\n- approach: tune\n- notes: none\n",
        encoding="utf-8",
    )


def _daily_text(root: Path) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8") for p in (root / "memory" / "daily").glob("*.md")
    )


def test_live_validation_blocks_smoke_after_kernel_failure(tmp_path: Path):
    from kaggle_agent.orchestrator import CycleResult

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    orchestrator = _judge_orch(root)
    result = CycleResult(
        competition="rsna-knee-abnormality-detection",
        dry_run=False,
        experiment_id="exp-1",
        kernel_ok=False,
        smoke_path=str(root / "smoke.csv"),
    )
    orchestrator._validate_sub(load_state(root), result)
    assert result.validate_ok is False
    assert result.candidate_csv is None
    assert "successful kernel" in result.errors[-1]


def test_validate_sub_kernel_judge_patches_experiment(tmp_path: Path):
    from kaggle_agent.config import load_competition
    from kaggle_agent.orchestrator import CycleResult

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    comp = load_competition("rsna_knee", root)
    orch = _judge_orch(root)
    exp_id = "20260815-000000"
    kernel_dir = _write_kernel_output(root, comp, exp_id)
    _write_exp(root, exp_id)
    from kaggle_agent.state_md import format_kv_markdown

    (root / "memory" / "kernel_job.md").write_text(
        format_kv_markdown(
            "kernel job",
            {"kernel_ref": "k", "folder": str(kernel_dir), "status": "success"},
        ),
        encoding="utf-8",
    )
    result = CycleResult(
        competition="rsna_knee", dry_run=True, kernel_path=str(kernel_dir), experiment_id=exp_id
    )
    orch._validate_sub(AgentState(), result)
    assert result.validate_ok is True
    assert result.kernel_judge_ok is True
    assert "judge kernel ready=True" in _daily_text(root)
    exp_text = (root / "memory" / "experiments" / f"{exp_id}.md").read_text(encoding="utf-8")
    assert "- judge: kernel True: " in exp_text


def test_validate_sub_image_template_requires_semantic_evidence(tmp_path: Path):
    from kaggle_agent.config import load_competition
    from kaggle_agent.orchestrator import CycleResult

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    comp = load_competition("rsna_knee", root)
    orch = _judge_orch(root)
    exp_id = "20260815-image-missing"
    kernel_dir = _write_kernel_output(root, comp, exp_id)
    (kernel_dir / "artifact_manifest.json").write_text(
        '{"template_version": "image-2d-dino-mil-v1"}',
        encoding="utf-8",
    )
    result = CycleResult(
        competition="rsna_knee", dry_run=True, kernel_path=str(kernel_dir), experiment_id=exp_id
    )

    orch._validate_sub(AgentState(), result)

    assert result.validate_ok is False
    assert "image semantic evidence missing" in result.errors[-1]


def test_validate_sub_image_template_accepts_semantic_evidence(tmp_path: Path):
    from kaggle_agent.config import load_competition
    from kaggle_agent.orchestrator import CycleResult

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    comp = load_competition("rsna_knee", root)
    orch = _judge_orch(root)
    exp_id = "20260815-image-ok"
    kernel_dir = _write_kernel_output(root, comp, exp_id)
    (kernel_dir / "artifact_manifest.json").write_text(
        '{"template_version": "image-2d-dino-mil-v1"}',
        encoding="utf-8",
    )
    (kernel_dir / "output" / "semantic_evidence.json").write_text(
        """
{
  "mounted_weights_loaded": true,
  "series_mapping_loaded": true,
  "mapped_series_count": 2,
  "mapped_study_count": 1,
  "decoded_non_empty_tensors": true,
  "report_labels_joined": true,
  "group_overlap": false,
  "optimizer_stepped": true,
  "checkpoints_written": true,
  "fold_predictions_written": true,
  "hidden_ids_from_folders": true,
  "submission_rows_match_hidden_ids": true,
  "resumed_folds": [0],
  "newly_trained_folds": [1, 2, 3, 4],
  "resume_checkpoint_source": "/kaggle/input/resume/fold_0_checkpoint.pt",
  "resume_checkpoint_sha256": "abc",
  "optimizer_steps": 4,
  "fold_outputs": ["fold_0_predictions.csv", "fold_1_predictions.csv", "fold_2_predictions.csv", "fold_3_predictions.csv", "fold_4_predictions.csv"],
  "prediction_hashes": ["h0", "h1", "h2", "h3", "h4"]
}
""",
        encoding="utf-8",
    )
    (kernel_dir / "output" / "artifact_manifest.runtime.json").write_text(
        '''
{
  "template_version": "image-2d-dino-mil-v1",
  "fold_outputs": ["fold_0_predictions.csv", "fold_1_predictions.csv", "fold_2_predictions.csv", "fold_3_predictions.csv", "fold_4_predictions.csv"],
  "prediction_hashes": ["h0", "h1", "h2", "h3", "h4"]
}
''',
        encoding="utf-8",
    )
    result = CycleResult(
        competition="rsna_knee", dry_run=True, kernel_path=str(kernel_dir), experiment_id=exp_id
    )

    orch._validate_sub(AgentState(), result)

    assert result.validate_ok is True
    events = "\n".join(
        p.read_text(encoding="utf-8") for p in (root / "memory" / "daily").glob("*.events.jsonl")
    )
    assert "image_semantic_evidence_ok" in events


def test_validate_sub_kernel_judge_rejects_failed_job(tmp_path: Path):
    from kaggle_agent.config import load_competition
    from kaggle_agent.orchestrator import CycleResult
    from kaggle_agent.state_md import format_kv_markdown

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    comp = load_competition("rsna_knee", root)
    orch = _judge_orch(root)
    exp_id = "20260815-000001"
    kernel_dir = _write_kernel_output(root, comp, exp_id)
    _write_exp(root, exp_id)
    (root / "memory" / "kernel_job.md").write_text(
        format_kv_markdown(
            "kernel job",
            {"kernel_ref": "k", "folder": str(kernel_dir), "status": "error"},
        ),
        encoding="utf-8",
    )
    result = CycleResult(
        competition="rsna_knee", dry_run=True, kernel_path=str(kernel_dir), experiment_id=exp_id
    )
    orch._validate_sub(AgentState(), result)
    assert result.validate_ok is True
    assert result.kernel_judge_ok is False
    assert "judge kernel ready=False" in _daily_text(root)
    exp_text = (root / "memory" / "experiments" / f"{exp_id}.md").read_text(encoding="utf-8")
    assert "- judge: kernel False: " in exp_text


def test_validate_sub_train_judge_flag_runs_llm(tmp_path: Path):
    import json

    from kaggle_agent.config import load_competition
    from kaggle_agent.orchestrator import CycleResult
    from kaggle_agent.state_md import format_kv_markdown

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    comp = load_competition("rsna_knee", root)

    class _ScriptedZen:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            self.calls += 1
            return json.dumps({"ready": True, "reason": "outputs plausible"})

    zen = _ScriptedZen()
    orch = _judge_orch(root, zen)
    orch.settings.raw["judges"] = {"train": True}
    exp_id = "20260815-000002"
    kernel_dir = _write_kernel_output(root, comp, exp_id)
    _write_exp(root, exp_id)
    (root / "memory" / "kernel_job.md").write_text(
        format_kv_markdown(
            "kernel job",
            {"kernel_ref": "k", "folder": str(kernel_dir), "status": "success"},
        ),
        encoding="utf-8",
    )
    result = CycleResult(
        competition="rsna_knee", dry_run=True, kernel_path=str(kernel_dir), experiment_id=exp_id
    )
    orch._validate_sub(AgentState(), result)
    assert result.kernel_judge_ok is True
    assert zen.calls == 1
    assert "judge train ready=True" in _daily_text(root)


def test_validate_sub_kernel_judge_skipped_without_kernel_output(tmp_path: Path):
    from kaggle_agent.orchestrator import CycleResult

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    orch = _judge_orch(root)
    exp_id = "20260815-000004"
    kernel_dir = root / "competitions" / "rsna_knee" / "notebooks" / exp_id
    (kernel_dir / "output").mkdir(parents=True)
    _write_exp(root, exp_id)
    result = CycleResult(
        competition="rsna_knee", dry_run=True, kernel_path=str(kernel_dir), experiment_id=exp_id
    )
    orch._validate_sub(AgentState(), result)
    assert result.validate_ok is False
    assert result.kernel_judge_ok is None
    assert "judge kernel" not in _daily_text(root)
    exp_text = (root / "memory" / "experiments" / f"{exp_id}.md").read_text(encoding="utf-8")
    assert "judge" not in exp_text


def test_validate_sub_logs_kernel_judge_skip_with_smoke_only(tmp_path: Path):
    from kaggle_agent.config import load_competition
    from kaggle_agent.orchestrator import CycleResult

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    orch = _judge_orch(root)
    comp = load_competition("rsna_knee", root)
    exp_id = "20260815-000005"
    kernel_dir = root / "competitions" / "rsna_knee" / "notebooks" / exp_id
    (kernel_dir / "output").mkdir(parents=True)
    _write_exp(root, exp_id)
    smoke_dir = root / "competitions" / "rsna_knee" / "submissions"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    header = ",".join([comp.id_column, *comp.labels])
    (smoke_dir / f"{exp_id}_smoke.csv").write_text(
        header
        + "\ns1,"
        + ",".join(["0.5"] * len(comp.labels))
        + "\ns2,"
        + ",".join(["0.5"] * len(comp.labels))
        + "\n",
        encoding="utf-8",
    )
    result = CycleResult(
        competition="rsna_knee", dry_run=True, kernel_path=str(kernel_dir), experiment_id=exp_id
    )
    orch._validate_sub(AgentState(), result)
    assert result.validate_ok is True
    assert result.kernel_judge_ok is None
    assert "judge kernel skipped: no kernel output" in _daily_text(root)
    exp_text = (root / "memory" / "experiments" / f"{exp_id}.md").read_text(encoding="utf-8")
    assert "judge" not in exp_text


def test_plan_judge_verdict_patched_to_experiment(tmp_path: Path):
    import json

    from kaggle_agent.orchestrator import CycleResult

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)

    class _ScriptedZen:
        def __init__(self) -> None:
            self.replies = [
                {
                    "tool": "write_plan",
                    "args": {
                        "hypothesis": "grouped cv",
                        "approach": "tune",
                        "steps": "grouped 5-fold CV over study IDs",
                    },
                },
                {"ready": True, "reason": "novel"},
                {"tool": "done", "args": {}},
            ]

        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            return json.dumps(self.replies.pop(0))

    orch = _judge_orch(root, _ScriptedZen())
    exp_id = "20260815-000003"
    result = CycleResult(competition="rsna_knee", dry_run=True, experiment_id=exp_id)
    orch._plan(AgentState(), dry=True, result=result)
    exp_text = (root / "memory" / "experiments" / f"{exp_id}.md").read_text(encoding="utf-8")
    assert "- judge: plan True: novel" in exp_text
    assert result.plan_text and "grouped 5-fold CV" in result.plan_text
