from kaggle_agent.autonomy.runtime import StageInput


def test_replay_epoch_invalidates_stage_without_using_git_sha():
    first = StageInput.create(stage="KERNEL_TRAIN", cycle_id="c", competition="demo", inputs={"x": 1}, replay_epoch=0)
    second = StageInput.create(stage="KERNEL_TRAIN", cycle_id="c", competition="demo", inputs={"x": 1}, replay_epoch=1)
    assert first.idempotency_key != second.idempotency_key
    assert "git" not in first.idempotency_key
