"""Run a production-shaped local worker/supervisor self-healing harness.

The harness creates a temporary git clone, adds a synthetic competition and a
single deterministic defect, then drives the real WorkerLauncher,
Supervisor, RepairCoordinator, generation promotion, and ResumeRequest path.
The only child-process hooks are test-fixture hooks for local external reads
and stage instrumentation; no production code is changed or bypassed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _run(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=root,
        text=True,
        capture_output=True,
        check=check,
    )


def _git(root: Path, *args: str) -> str:
    return _run(root, "git", *args).stdout.strip()


def _write_fixture_files(source: Path) -> None:
    import yaml

    settings_path = source / "config" / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    settings["default_competition"] = "full_system_harness"
    settings["browser_research"]["enabled"] = False
    settings["research"]["deep"]["enabled"] = False
    settings["kernel"]["push"] = False
    settings["orchestrator"]["dry_run"] = True
    settings["supervisor"].update({"enabled": True, "mode": "auto_safe"})
    settings["supervisor"]["promotion"]["automatic"] = True
    settings["supervisor"]["auto_safe"]["enabled"] = True
    settings["supervisor"]["repair"].update(
        {
            "max_attempts_per_incident": 2,
            "max_repairs_per_cycle": 1,
            "max_repairs_per_day": 1,
            "max_changed_source_files": 4,
            "max_changed_test_files": 2,
            "max_changed_lines": 250,
        }
    )
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    config_dir = source / "config" / "competitions"
    config_text = (config_dir / "rsna_knee.yaml").read_text(encoding="utf-8")
    config_text = config_text.replace("id: rsna_knee", "id: full_system_harness")
    config_text = config_text.replace(
        "slug: rsna-knee-abnormality-detection",
        "slug: local-full-system-harness",
    )
    config_text = config_text.replace(
        "workspace: \n  relative: competitions/rsna_knee",
        "workspace:\n  relative: competitions/full_system_harness",
    )
    config_text = config_text.replace(
        "relative: competitions/rsna_knee",
        "relative: competitions/full_system_harness",
    )
    config_text = config_text.replace(
        "title: RSNA Knee Abnormality Detection",
        "title: Local Full System Harness",
    )
    config_text = config_text.replace(
        "fleet: [notebooks, papers, github, web, discussions, datasets]",
        "fleet: false",
    )
    (config_dir / "full_system_harness.yaml").write_text(config_text, encoding="utf-8")

    shutil.copytree(
        source / "competitions" / "rsna_knee",
        source / "competitions" / "full_system_harness",
        ignore=shutil.ignore_patterns("data", "notebooks", "submissions", "research-cache"),
    )
    pipeline = source / "competitions" / "full_system_harness" / "pipeline"
    methods = {
        "dataset_sources": [],
        "model_sources": [],
        "infer_hints": ["use the local synthetic submission schema"],
        "implement_steps": ["run the local synthetic smoke parser"],
        "source_card_refs": ["source-harness"],
    }
    (pipeline / "methods.json").write_text(json.dumps(methods, indent=2) + "\n", encoding="utf-8")
    (pipeline / "defect.py").write_text(
        "def trigger() -> str:\n"
        "    expected = 'harness-ok'\n"
        "    return resultz  # deliberate LOW-risk NameError\n",
        encoding="utf-8",
    )

    data = source / "data"
    data.mkdir(parents=True, exist_ok=True)
    labels = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]
    header = ",".join(["StudyInstanceUID", *labels])
    rows = [
        "harness-1," + ",".join(["0.5"] * len(labels)),
        "harness-2," + ",".join(["0.6"] * len(labels)),
        "harness-3," + ",".join(["0.4"] * len(labels)),
    ]
    (data / "sample_submission.csv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    (data / "test.csv").write_text(
        "StudyInstanceUID\n" + "\n".join(["harness-1", "harness-2", "harness-3"]) + "\n",
        encoding="utf-8",
    )

    test_body = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "for parent in Path(__file__).resolve().parents:\n"
        "    if (parent / 'competitions').is_dir():\n"
        "        sys.path.insert(0, str(parent))\n"
        "        break\n"
        "from competitions.full_system_harness.pipeline.defect import trigger\n\n\n"
        "def test_trigger_returns_expected_marker():\n"
        "    assert trigger() == 'harness-ok'\n"
    )
    competition_test = source / "competitions" / "full_system_harness" / "tests"
    competition_test.mkdir(parents=True, exist_ok=True)
    (competition_test / "test_defect.py").write_text(test_body, encoding="utf-8")
    (pipeline / "test_defect.py").write_text(test_body, encoding="utf-8")
    # Keep the conventional repository-level focused-test alias available as
    # well. DeepSeek may choose either narrow path; neither test is writable by
    # the repair spec for this existing-test defect.
    root_test = source / "tests"
    root_test.mkdir(parents=True, exist_ok=True)
    (root_test / "test_pipeline_defect.py").write_text(test_body, encoding="utf-8")
    (root_test / "test_defect.py").write_text(test_body, encoding="utf-8")
    (root_test / "harness" / "test_defect.py").parent.mkdir(parents=True, exist_ok=True)
    (root_test / "harness" / "test_defect.py").write_text(test_body, encoding="utf-8")
    (source / "memory" / "templates" / "FULL_SYSTEM_HARNESS.md").write_text(
        "Synthetic local competition fixture; no external mutations are allowed.\n",
        encoding="utf-8",
    )
    sitecustomize = source / "sitecustomize.py"
    sitecustomize.write_text(
        '''"""Child-process-only hooks for scripts/run_full_system_harness.py."""\n\n'''
        "from __future__ import annotations\n\n"
        "import importlib.util\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "if any(\"pytest\" in argument for argument in sys.argv):\n"
        "    os.environ.pop(\"KAGGLE_AGENT_STATE_ROOT\", None)\n"
        "    os.environ.pop(\"KAGGLE_AGENT_SUPERVISOR_DIR\", None)\n"
        "    os.environ.pop(\"KAGGLE_AGENT_FULL_SYSTEM_HARNESS\", None)\n\n"
        "if os.environ.get(\"KAGGLE_AGENT_FULL_SYSTEM_HARNESS\") == \"1\":\n"
        "    from kaggle_agent.orchestrator import Orchestrator\n\n"
        "    from kaggle_agent.llm.router import ModelRouter\n\n"
        "    def _offline_router(cls, settings, competition):\n"
        "        return cls(settings=settings, competition=competition, client=None)\n\n"
        "    ModelRouter.build = classmethod(_offline_router)\n\n"
        "    def _count(stage: str) -> None:\n"
        "        root = Path(os.environ[\"KAGGLE_AGENT_STATE_ROOT\"])\n"
        "        path = root / \"harness-stage-calls.json\"\n"
        "        try:\n"
        "            values = json.loads(path.read_text(encoding=\"utf-8\")) if path.is_file() else {}\n"
        "        except json.JSONDecodeError:\n"
        "            values = {}\n"
        "        values[stage] = int(values.get(stage, 0)) + 1\n"
        "        temporary = path.with_suffix(\".tmp\")\n"
        "        temporary.write_text(json.dumps(values, sort_keys=True) + \"\\n\", encoding=\"utf-8\")\n"
        "        temporary.replace(path)\n\n"
        "    def _research(self, state, result):\n"
        "        _count(\"RESEARCH\")\n"
        "        return _ORIGINAL_RESEARCH(self, state, result)\n\n"
        "    def _plan(self, state, dry, result):\n"
        "        _count(\"PLAN\")\n"
        "        return _ORIGINAL_PLAN(self, state, dry, result)\n\n"
        "    def _code(self, state, result):\n"
        "        _count(\"CODE\")\n"
        "        if self._resume_request is None:\n"
        "            marker = Path(os.environ[\"KAGGLE_AGENT_STATE_ROOT\"]) / \"harness-defect-triggered\"\n"
        "            if not marker.exists():\n"
        "                marker.write_text(\"triggered\\n\", encoding=\"utf-8\")\n"
        "                defect_path = self.root / \"competitions/full_system_harness/pipeline/defect.py\"\n"
        "                module_spec = importlib.util.spec_from_file_location(\"harness_defect\", defect_path)\n"
        "                module = importlib.util.module_from_spec(module_spec)\n"
        "                assert module_spec and module_spec.loader\n"
        "                module_spec.loader.exec_module(module)\n"
        "                try:\n"
        "                    module.trigger()\n"
        "                except Exception as exc:\n"
        "                    result.code_ok = False\n"
        "                    result.errors.append(\"code: %s in competitions/full_system_harness/pipeline/defect.py: %s; existing focused test: tests/test_defect.py\" % (type(exc).__name__, exc))\n"
        "                    return state\n"
        "        return _ORIGINAL_CODE(self, state, result)\n\n"
        "    def _kaggle_snapshot(self, state, result):\n"
        "        result.kaggle_ok = True\n\n"
        "    def _browser_research(self, result):\n"
        "        result.browser_ok = True\n\n"
        "    def _deep_research(self, result, **kwargs):\n"
        "        result.deep_ok = True\n\n"
        "    _ORIGINAL_RESEARCH = Orchestrator._research\n"
        "    _ORIGINAL_PLAN = Orchestrator._plan\n"
        "    _ORIGINAL_CODE = Orchestrator._code\n"
        "    Orchestrator._research = _research\n"
        "    Orchestrator._plan = _plan\n"
        "    Orchestrator._code = _code\n"
        "    Orchestrator._kaggle_snapshot = _kaggle_snapshot\n"
        "    Orchestrator._browser_research = _browser_research\n"
        "    Orchestrator._deep_research = _deep_research\n",
        encoding="utf-8",
    )
    # WorkerLauncher prepends generation/src to PYTHONPATH.  Python imports
    # sitecustomize during startup before reliably searching the cwd, so keep
    # the same fixture hook on that explicit launcher path as well.
    (source / "src" / "sitecustomize.py").write_text(
        sitecustomize.read_text(encoding="utf-8"), encoding="utf-8"
    )


def _prepare_runtime(source: Path, state_root: Path) -> None:
    from kaggle_agent.state_md import AgentState, save_state
    from kaggle_agent.loop import LoopState, save_loop

    memory = state_root / "memory"
    (memory / "research-deep").mkdir(parents=True, exist_ok=True)
    (memory / "experiments").mkdir(parents=True, exist_ok=True)
    (memory / "daily").mkdir(parents=True, exist_ok=True)
    templates = source / "memory" / "templates"
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        source_path = templates / name
        target = memory / name
        if source_path.is_file():
            shutil.copy2(source_path, target)
        else:
            target.write_text("# synthetic harness\n", encoding="utf-8")
    (memory / "COMPETITION.md").write_text(
        "# Full System Harness\n\nSynthetic local-only competition.\n", encoding="utf-8"
    )
    (memory / "research.md").write_text(
        "# Research\n\n## Method cards\n\nSynthetic local card.\n", encoding="utf-8"
    )
    (memory / "research-deep" / "source-harness.md").write_text(
        "# Synthetic harness card\n"
        "- ref: local/harness\n"
        "- copyable next step: run the local synthetic smoke parser\n"
        "- do not copy: external mutation\n"
        "- datasets_mentioned: none\n"
        "- models_mentioned: none\n",
        encoding="utf-8",
    )
    save_state(AgentState(paused=False, competition="full_system_harness"), state_root)
    save_loop(LoopState(next_n="1"), state_root)


def _summarize(state_root: Path, first: Any) -> dict[str, Any]:
    from kaggle_agent.supervisor.generation import GenerationStore
    from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore

    store = SupervisorStateStore(RuntimeLayout.for_repo(state_root, state_root))
    promotion = store.read_json("promotion.json", {}) or {}
    incidents = list((state_root / "incidents").glob("*.json"))
    accepted = []
    accepted_repairs = []
    for acceptance_path in (state_root / "repairs").glob("*/acceptance.json"):
        acceptance = store.read_json(str(acceptance_path.relative_to(state_root)), {}) or {}
        fields = acceptance.get("acceptance") or {}
        if fields and all(bool(value) for value in fields.values()):
            accepted.append(acceptance_path)
            generation = acceptance.get("generation") or {}
            accepted_repairs.append(
                {
                    "repair_id": generation.get("repair_id"),
                    "candidate_sha": (generation.get("revision") or {}).get("git_sha"),
                }
            )
    generations = list((state_root / "generations").glob("generation-*.json"))
    workers = []
    for path in sorted((state_root / "workers").glob("*/metadata.json")):
        row = store.read_json(str(path.relative_to(state_root)), {}) or {}
        workers.append({"worker_id": row.get("worker_id"), "pid": row.get("pid"), "generation_id": row.get("generation_id")})
    calls = store.read_json("harness-stage-calls.json", {}) or {}
    ledger = state_root / ".agent" / "stage-ledger.jsonl"
    ledger_rows = []
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "stage_finished":
                ledger_rows.append({"stage": row.get("stage"), "state": row.get("state"), "replay_epoch": row.get("replay_epoch")})
    generation_ids = []
    generation_store = GenerationStore(store)
    for path in generations:
        generation = generation_store.load(path.stem)
        if generation:
            generation_ids.append(generation.generation_id)
    return {
        "first_supervisor_status": getattr(first, "status", str(first)),
        "first_supervisor_reason": getattr(first, "reason", ""),
        "incident_count": len(incidents),
        "accepted_repair_count": len(accepted),
        "accepted_repairs": accepted_repairs,
        "generation_count": len(generations),
        "generation_ids": generation_ids,
        "promotion": promotion,
        "workers": workers,
        "stage_calls": calls,
        "stage_ledger_finished": ledger_rows,
        "kaggle_mutations": 0,
        "telegram_messages": 0,
        "active_generation": (store.read_json("active-generation.json", {}) or {}).get("generation_id"),
    }


def _run_supervisor_subprocess(source: Path, state_root: Path) -> subprocess.CompletedProcess[str]:
    """Restart a real supervisor process from only durable fixture state."""
    environment = os.environ.copy()
    environment.update(
        {
            "KAGGLE_AGENT_SUPERVISOR_DIR": str(state_root),
            "KAGGLE_AGENT_STATE_ROOT": str(state_root),
            "KAGGLE_AGENT_FULL_SYSTEM_HARNESS": "1",
            "PYTHONPATH": str(source / "src"),
            "UV_PROJECT_ENVIRONMENT": str(Path.cwd() / ".venv"),
        }
    )
    code = (
        "from pathlib import Path; "
        "from kaggle_agent.config import load_settings; "
        "from kaggle_agent.supervisor.loop import Supervisor; "
        f"root=Path({str(source)!r}); s=Supervisor(load_settings(root), root); "
        "r=s.run_once(competition='full_system_harness', wait=True); "
        "print(r.status); raise SystemExit(0 if r.status == 'SUCCESS' else 1)"
    )
    return subprocess.run(
        (sys.executable, "-c", code),
        cwd=source,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_recovery_probes(source: Path, state_root: Path) -> dict[str, Any]:
    """Exercise restart and rollback using real supervisor/worker processes."""
    from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore

    store = SupervisorStateStore(RuntimeLayout.for_repo(state_root, state_root))
    transaction = store.read_json("promotion.json", {}) or {}
    replacement_worker = str(transaction["replacement_worker_id"])
    result_path = store.path(f"workers/{replacement_worker}/result.json")
    result_path.unlink()
    metadata = store.read_json(f"workers/{replacement_worker}/metadata.json", {}) or {}
    metadata.update({"pid": 99999999, "launch_state": "STARTED"})
    store.write_json(f"workers/{replacement_worker}/metadata.json", metadata)
    store.write_json("promotion.json", {**transaction, "status": "PROMOTED"})

    restart = _run_supervisor_subprocess(source, state_root)
    after_restart = store.read_json("promotion.json", {}) or {}
    restart_pass = (
        restart.returncode == 0
        and restart.stdout.strip().endswith("SUCCESS")
        and after_restart.get("status") == "RESUMED"
        and after_restart.get("resumed_worker_id") == replacement_worker
        and (store.read_json("active-generation.json", {}) or {}).get("generation_id") == "generation-0002"
    )

    rollback_worker = "worker-unhealthy-replacement"
    rollback_transaction = {
        **after_restart,
        "status": "PROMOTED",
        "replacement_worker_id": rollback_worker,
    }
    store.write_json("promotion.json", rollback_transaction)
    store.write_json(
        f"workers/{rollback_worker}/metadata.json",
        {
            "pid": None,
            "worker_id": rollback_worker,
            "generation_id": "generation-0002",
            "supervisor_token": "recovery-probe",
            "launch_state": "STARTED",
        },
    )
    store.write_json(
        f"workers/{rollback_worker}/result.json",
        {"status": "FATAL", "exit_reason": "synthetic startup import failure"},
    )
    rollback = _run_supervisor_subprocess(source, state_root)
    after_rollback = store.read_json("promotion.json", {}) or {}
    rollback_pass = (
        rollback.returncode != 0
        and rollback.stdout.strip().endswith("ROLLED_BACK")
        and after_rollback.get("status") == "ROLLED_BACK"
        and (store.read_json("active-generation.json", {}) or {}).get("generation_id") == "generation-0001"
    )
    return {
        "restart_after_promoted_before_result": {
            "supervisor_exit": restart.returncode,
            "worker_result": restart.stdout.strip(),
            "status": after_restart.get("status"),
            "passed": restart_pass,
        },
        "unhealthy_replacement_rollback": {
            "supervisor_exit": rollback.returncode,
            "worker_result": rollback.stdout.strip(),
            "status": after_rollback.get("status"),
            "active_generation": (store.read_json("active-generation.json", {}) or {}).get("generation_id"),
            "passed": rollback_pass,
        },
    }


def run(*, keep: bool = False) -> int:
    from kaggle_agent.supervisor.agents import DeepSeekSupervisorAgents
    from kaggle_agent.supervisor.loop import Supervisor
    from kaggle_agent.config import load_settings

    if DeepSeekSupervisorAgents.from_env() is None:
        print("BLOCKED: DeepSeek credential unavailable in environment")
        return 2
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="kaggle-agent-full-system-") as temp_dir:
        temp = Path(temp_dir)
        source = temp / "source"
        subprocess.run(("git", "clone", "--no-local", str(repo), str(source)), check=True, capture_output=True, text=True)
        _write_fixture_files(source)
        _run(source, "git", "config", "user.email", "full-system-harness@example.invalid")
        _run(source, "git", "config", "user.name", "full-system harness")
        _run(source, "git", "add", "-A")
        _run(source, "git", "add", "-f", "data/sample_submission.csv", "data/test.csv")
        _run(source, "git", "commit", "-m", "test: add disposable full-system harness fixture")

        state_root = temp / "state"
        _prepare_runtime(source, state_root)
        old_env = os.environ.copy()
        os.environ.update(
            {
                "KAGGLE_AGENT_SUPERVISOR_DIR": str(state_root),
                "KAGGLE_AGENT_STATE_ROOT": str(state_root),
                "KAGGLE_AGENT_FULL_SYSTEM_HARNESS": "1",
                "UV_PROJECT_ENVIRONMENT": str(Path.cwd() / ".venv"),
            }
        )
        try:
            settings = load_settings(source)
            supervisor = Supervisor(settings, source)
            first = supervisor.run_once(competition="full_system_harness", wait=True)
            summary = _summarize(state_root, first)
            recovery = _run_recovery_probes(source, state_root)
            summary["recovery"] = recovery
            print(json.dumps(summary, indent=2, sort_keys=True))
            required = {
                "first_supervisor_status": "SUCCESS",
                "incident_count": 1,
                "accepted_repair_count": 1,
                "generation_count": 2,
                "active_generation": "generation-0002",
                "stage_calls": {"RESEARCH": 1, "PLAN": 1, "CODE": 2},
            }
            passed = all(summary.get(key) == value for key, value in required.items())
            passed = passed and summary.get("promotion", {}).get("status") == "RESUMED"
            passed = passed and len(summary.get("workers", [])) == 2
            passed = passed and all(item.get("passed") for item in recovery.values())
            print("FULL_SYSTEM_HARNESS: PASS" if passed else "FULL_SYSTEM_HARNESS: FAIL")
            if keep:
                keep_path = repo / ".full-system-harness-last-run"
                if keep_path.exists():
                    shutil.rmtree(keep_path)
                shutil.copytree(temp, keep_path)
                print(f"HARNESS_ARTIFACTS: {keep_path}")
            return 0 if passed else 1
        finally:
            os.environ.clear()
            os.environ.update(old_env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="retain disposable artifacts under the repository")
    args = parser.parse_args(argv)
    return run(keep=args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
