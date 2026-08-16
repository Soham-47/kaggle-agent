from pathlib import Path

from kaggle_agent.train.kernel_history import (
    package_fingerprint,
    seen_recipe,
    record_kernel,
    seen_kernel,
)


def test_kernel_history_detects_same_package(tmp_path: Path):
    folder = tmp_path / "package"
    folder.mkdir()
    (folder / "agent_baseline.ipynb").write_text("notebook", encoding="utf-8")
    (folder / "kernel-metadata.json").write_text("{}", encoding="utf-8")

    fingerprint = package_fingerprint(folder)
    assert seen_kernel(tmp_path, fingerprint) is False
    record_kernel(tmp_path, "owner/kernel", fingerprint)
    assert seen_kernel(tmp_path, fingerprint) is True


def test_package_fingerprint_covers_bundled_files(tmp_path: Path):
    folder = tmp_path / "package"
    folder.mkdir()
    (folder / "agent_baseline.ipynb").write_text("notebook", encoding="utf-8")
    (folder / "kernel-metadata.json").write_text("{}", encoding="utf-8")
    (folder / "methods.json").write_text("first", encoding="utf-8")
    first = package_fingerprint(folder)

    (folder / "methods.json").write_text("second", encoding="utf-8")

    assert package_fingerprint(folder) != first


def test_seen_recipe_reads_old_package_manifests(tmp_path: Path):
    folder = tmp_path / "competitions" / "demo" / "notebooks" / "exp"
    folder.mkdir(parents=True)
    (folder / "kernel-metadata.json").write_text(
        '{"id":"owner/kernel","experiment_manifest":{"recipe_sha256":"abc"}}',
        encoding="utf-8",
    )
    record_kernel(tmp_path, "owner/kernel", "package", "")

    assert seen_recipe(tmp_path, "abc") is True
