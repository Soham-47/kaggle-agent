"""Shared tmp workspace setup for cycle tests."""

from __future__ import annotations

from pathlib import Path

from kaggle_agent.loop import LoopState, save_loop
from kaggle_agent.state_md import AgentState, save_state


def write_min_study_csv(root: Path) -> Path:
    """Tmp fixtures do not include study IDs; kernel package needs some."""
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    path = data / "sample_submission.csv"
    path.write_text("StudyInstanceUID\ns1\ns2\n", encoding="utf-8")
    return path


def write_kernel_fixture_data(root: Path) -> Path:
    """Create small public-shaped tables for kernel and ranker tests.

    Tests must exercise the file contracts without depending on ignored
    developer-local Kaggle data. The values are synthetic and contain no
    competition records.
    """
    import csv

    labels = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    test_ids = ["test-a", "test-b", "test-c"]
    with (data / "sample_submission.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["StudyInstanceUID", *labels])
        writer.writeheader()
        for study_id in test_ids:
            writer.writerow({"StudyInstanceUID": study_id, **{label: 0.5 for label in labels}})
    with (data / "test.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["StudyInstanceUID"])
        writer.writeheader()
        writer.writerows({"StudyInstanceUID": study_id} for study_id in test_ids)
    with (data / "test_series.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"],
        )
        writer.writeheader()
        for index, study_id in enumerate(test_ids):
            writer.writerow({
                "StudyInstanceUID": study_id,
                "SeriesInstanceUID": f"series-{index}",
                "Anatomical_Plane": "Axial" if index % 2 else "Sagittal",
                "Fluid_Sensitive": index % 2,
                "Fat_Suppression": (index + 1) % 2,
            })
    train_ids = [f"train-{index}" for index in range(60)]
    with (data / "train.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["StudyInstanceUID", *labels])
        writer.writeheader()
        for index, study_id in enumerate(train_ids):
            writer.writerow({"StudyInstanceUID": study_id, **{label: (index + offset) % 2 for offset, label in enumerate(labels)}})
    with (data / "train_series.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"],
        )
        writer.writeheader()
        for index, study_id in enumerate(train_ids):
            writer.writerow({
                "StudyInstanceUID": study_id,
                "SeriesInstanceUID": f"train-series-{index}",
                "Anatomical_Plane": "Axial" if index % 2 else "Sagittal",
                "Fluid_Sensitive": index % 2,
                "Fat_Suppression": (index + 1) % 2,
            })
    return data


def copy_min_workspace(
    root: Path, real: Path, *, competition: str = "rsna_knee"
) -> None:
    import shutil

    shutil.copytree(real / "config", root / "config")
    skip = {
        "notebooks",
        "submissions",
        "research-cache",
        "experiments",
        "data",
        "__pycache__",
    }
    shutil.copytree(
        real / "competitions",
        root / "competitions",
        ignore=lambda _dir, names: [n for n in names if n in skip],
    )
    # These cycle tests target train/submit control flow. Keep the optional
    # source-agent fleet out of their fixture so missing model credentials do
    # not turn a submit assertion into a research-verification assertion.
    competition_config = root / "config" / "competitions" / "rsna_knee.yaml"
    text = competition_config.read_text(encoding="utf-8")
    text = text.replace(
        "fleet: [notebooks, papers, github, web, discussions, datasets]",
        "fleet: false",
    )
    competition_config.write_text(text, encoding="utf-8")
    (root / "memory").mkdir()
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        shutil.copy(real / "memory" / "templates" / name, root / "memory" / name)
    (root / "memory" / "experiments").mkdir()
    (root / "memory" / "daily").mkdir()
    write_min_study_csv(root)
    # Real state may be paused; tests need a clean agent
    save_state(AgentState(paused=False, competition=competition), root)
    # Keep existing cycle tests at N=1; production missing loop.md still defaults to 3
    save_loop(LoopState(next_n="1"), root)
