from pathlib import Path

from kaggle_agent.autonomy.onboard import OnboardResult
from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent import cli


def test_cli_onboard_accepts_slug_and_reports_contract(monkeypatch, capsys, tmp_path: Path):
    contract = type("Contract", (), {"raw": {"id": "demo"}, "compatibility_hash": "abc"})()
    monkeypatch.setattr(cli, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_onboard_slug", lambda root, slug: OnboardResult(
        StageOutcome.success("BOOTSTRAP", f"verified {slug}"), contract
    ))
    assert cli.main(["onboard", "demo-slug"]) == 0
    output = capsys.readouterr().out
    assert "competition=demo" in output
    assert "contract_hash=abc" in output
