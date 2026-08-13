"""Live Kaggle API checks — skipped unless credentials work."""

from __future__ import annotations

import pytest

from kaggle_agent.kaggle_api import KaggleApiError, KaggleClient

SLUG = "rsna-knee-abnormality-detection"


@pytest.mark.integration
def test_live_limits_and_leaderboard():
    try:
        client = KaggleClient().connect()
    except KaggleApiError as exc:
        pytest.skip(f"no kaggle auth: {exc}")

    lim = client.submission_limits(SLUG)
    assert lim.num_allowed_now >= 0

    lb = client.leaderboard(SLUG, top=3)
    assert len(lb) >= 1
    assert lb[0].score

    meta = client.list_meta_files(SLUG, max_pages=1)
    names = {f.name for f in meta}
    assert "sample_submission.csv" in names
