from pathlib import Path
from types import SimpleNamespace

import yaml

from kaggle_agent.autonomy.onboard import CompetitionBootstrapper
from kaggle_agent.autonomy.outcomes import OutcomeState
from kaggle_agent.kaggle_api.client import KaggleClient
from kaggle_agent.state_md import load_state
from kaggle_agent.config import load_competition
from kaggle_agent.code.workspace import ensure_pipeline_ready
from fakes import FakeKaggleApi


class OnboardApi(FakeKaggleApi):
    def competitions_list(self, **kwargs):
        return SimpleNamespace(
            competitions=[
                SimpleNamespace(
                    ref="https://www.kaggle.com/competitions/demo-slug",
                    title="Demo Classification",
                    url="https://www.kaggle.com/competitions/demo-slug",
                    deadline="2027-01-01T00:00:00Z",
                    evaluationMetric="Roc Auc Score",
                    isKernelsSubmissionsOnly=False,
                    maxDailySubmissions=5,
                    tags=[
                        SimpleNamespace(name="tabular"),
                        SimpleNamespace(name="binary classification"),
                    ],
                )
            ]
        )

    def competition_download_file(self, competition, file_name, path=None, **kwargs):
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        if file_name == "sample_submission.csv":
            (dest / file_name).write_text("id,target\na,0.5\nb,0.5\n", encoding="utf-8")
        elif file_name == "train.csv":
            (dest / file_name).write_text("id,x,target\na,1,0\nb,2,1\n", encoding="utf-8")

    def competition_list_files(self, competition, page_token=None, page_size=20):
        files = [
            SimpleNamespace(name="sample_submission.csv", total_bytes=40, ref="sample_submission.csv"),
            SimpleNamespace(name="train.csv", total_bytes=40, ref="train.csv"),
            SimpleNamespace(name="test.csv", total_bytes=30, ref="test.csv"),
        ]
        return SimpleNamespace(files=files, next_page_token=None)


def _root(tmp_path: Path) -> Path:
    (tmp_path / "config" / "competitions").mkdir(parents=True)
    (tmp_path / "config" / "settings.yaml").write_text(
        "default_competition: old\n", encoding="utf-8"
    )
    (tmp_path / "memory").mkdir()
    return tmp_path


def test_slug_only_onboarding_writes_verified_contract_and_activates(tmp_path: Path):
    root = _root(tmp_path)
    client = KaggleClient(api=OnboardApi()).connect()
    result = CompetitionBootstrapper(root, client).onboard("demo-slug")
    assert result.outcome.state is OutcomeState.SUCCESS
    config = yaml.safe_load((root / "config/competitions/demo_slug.yaml").read_text())
    assert config["task"]["family"] == "tabular_classification"
    assert config["submission"]["columns"] == ["id", "target"]
    assert config["data"]["target_columns"] == ["target"]
    assert config["contract_hash"] == result.contract.compatibility_hash
    assert yaml.safe_load((root / "config/settings.yaml").read_text())["default_competition"] == "demo_slug"
    assert load_state(root).competition == "demo_slug"
    assert (root / "competitions/demo_slug/pipeline").is_dir()
    loaded = load_competition("demo_slug", root)
    assert loaded.id_column == "id"
    assert loaded.labels == ["target"]
    assert ensure_pipeline_ready(root / loaded.workspace_relative).ok
    assert (root / "competitions/demo_slug/data/sample_submission.csv").is_file()


def test_onboarding_does_not_mutate_repo_when_task_is_ambiguous(tmp_path: Path):
    root = _root(tmp_path)
    api = OnboardApi()
    api.competitions_list = lambda **kwargs: SimpleNamespace(
        competitions=[SimpleNamespace(
            ref="demo-slug", title="Mystery", url="u", deadline="d",
            evaluationMetric="Custom Metric", isKernelsSubmissionsOnly=False,
            maxDailySubmissions=1, tags=[]
        )]
    )
    result = CompetitionBootstrapper(root, KaggleClient(api=api).connect()).onboard("demo-slug")
    assert result.outcome.state is OutcomeState.NEEDS_AUTHORITY
    assert not (root / "config/competitions/demo_slug.yaml").exists()
    assert yaml.safe_load((root / "config/settings.yaml").read_text())["default_competition"] == "old"
