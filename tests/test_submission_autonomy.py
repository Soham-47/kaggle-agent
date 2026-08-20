from pathlib import Path

import yaml

from kaggle_agent.autonomy.approval import SubmissionAutonomy


def _config(path: Path):
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({
        "id": "demo",
        "contract_hash": "contract-a",
        "autonomy": {
            "first_submission_approved": False,
            "approved_contract_hash": None,
            "max_submissions_per_day": 2,
        },
    }), encoding="utf-8")


def test_first_success_persists_competition_scoped_autonomy(tmp_path: Path):
    path = tmp_path / "config/competitions/demo.yaml"
    _config(path)
    policy = SubmissionAutonomy(path)
    assert policy.can_submit_without_approval(proposals_used=0) is False
    policy.record_approved_submission()
    assert policy.can_submit_without_approval(proposals_used=1) is True
    raw = yaml.safe_load(path.read_text())
    assert raw["autonomy"]["approved_contract_hash"] == "contract-a"


def test_contract_change_resets_automatic_submission(tmp_path: Path):
    path = tmp_path / "config/competitions/demo.yaml"
    _config(path)
    policy = SubmissionAutonomy(path)
    policy.record_approved_submission()
    raw = yaml.safe_load(path.read_text())
    raw["contract_hash"] = "contract-b"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert SubmissionAutonomy(path).can_submit_without_approval(proposals_used=0) is False


def test_daily_budget_blocks_automatic_submission(tmp_path: Path):
    path = tmp_path / "config/competitions/demo.yaml"
    _config(path)
    policy = SubmissionAutonomy(path)
    policy.record_approved_submission()
    assert policy.can_submit_without_approval(proposals_used=2) is False
