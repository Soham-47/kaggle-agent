"""Unit tests for KaggleClient with an injected fake API (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_agent.kaggle_api import KaggleApiError
from kaggle_agent.kaggle_api.client import KaggleClient
from kaggle_agent.kaggle_api.models import ResearchSnapshot
from kaggle_agent.research.apply_snapshot import apply_kaggle_research
from kaggle_agent.state_md import load_state
from fakes import FakeKaggleApi


@pytest.fixture
def client() -> KaggleClient:
    return KaggleClient(api=FakeKaggleApi()).connect()


def test_connect_authenticates():
    api = FakeKaggleApi()
    KaggleClient(api=api).connect()
    assert api.authenticated is True


def test_submission_limits(client: KaggleClient):
    lim = client.submission_limits("rsna-knee-abnormality-detection")
    assert lim.num_allowed_now == 4
    assert lim.can_submit is True


def test_list_meta_files_skips_dicom_paths(client: KaggleClient):
    files = client.list_meta_files("rsna-knee-abnormality-detection")
    names = {f.name for f in files}
    assert "sample_submission.csv" in names
    assert "test.csv" in names
    assert not any(".dcm" in n for n in names)


def test_leaderboard_and_kernels(client: KaggleClient):
    lb = client.leaderboard("rsna-knee", top=2)
    assert lb[0].team_name == "Alpha"
    assert lb[0].score == "0.94"
    kn = client.kernels("rsna-knee", top=1)
    assert kn[0].ref == "user/baseline"
    assert "kaggle.com/code" in kn[0].url


def test_submit_dry_run_does_not_call_api(client: KaggleClient, tmp_path: Path):
    csv = tmp_path / "s.csv"
    csv.write_text("a\n", encoding="utf-8")
    r = client.submit("rsna-knee", csv, "msg", dry_run=True)
    assert r.dry_run is True
    assert r.success is True
    assert client.api.submit_calls == []  # type: ignore[attr-defined]


def test_submit_live_calls_api(client: KaggleClient, tmp_path: Path):
    csv = tmp_path / "s.csv"
    csv.write_text("a\n", encoding="utf-8")
    r = client.submit("rsna-knee", csv, "msg", dry_run=False)
    assert r.success is True
    assert r.dry_run is False
    assert len(client.api.submit_calls) == 1  # type: ignore[attr-defined]


def test_download_rejects_nested_path(client: KaggleClient, tmp_path: Path):
    with pytest.raises(KaggleApiError):
        client.download_file("rsna", "test_series/x.dcm", tmp_path)


def test_download_file(client: KaggleClient, tmp_path: Path):
    path = client.download_file("rsna", "sample_submission.csv", tmp_path)
    assert path.is_file()
    assert "StudyInstanceUID" in path.read_text(encoding="utf-8")


def test_research_snapshot_and_apply(client: KaggleClient, tmp_path: Path):
    snap = client.research_snapshot("rsna-knee-abnormality-detection")
    assert isinstance(snap, ResearchSnapshot)
    assert snap.limits and snap.limits.num_allowed_now == 4
    assert snap.meta_files
    assert snap.leaderboard
    assert snap.kernels
    assert not snap.errors

    (tmp_path / "memory").mkdir()
    apply_kaggle_research(snap, tmp_path, agent_max_proposals=2)
    research = (tmp_path / "memory" / "research.md").read_text(encoding="utf-8")
    assert "allowed_now: 4" in research
    assert "Alpha" in research
    assert "Baseline CNN" in research
    st = load_state(tmp_path)
    assert st.max_proposals == "2"
    assert "kaggle_allowed_now=4" in st.note
    assert st.budget_date
