import pytest

from shared.batching import SimContinuousBatcher


def test_single_request_lifecycle():
    b = SimContinuousBatcher(max_batch=2, max_seq_len=10, kv_per_token=1)
    assert b.submit(0.0, 3) == 0
    assert b.step(1) == []
    assert b.step(2) == []
    assert b.step(3) == [0]


def test_stats_after_single_request():
    b = SimContinuousBatcher(2, 10, 1)
    b.submit(0.0, 3)
    b.step(1)
    b.step(2)
    b.step(3)
    assert b.stats() == {
        "mean_ttft": pytest.approx(1.0),
        "mean_tbt": pytest.approx(1.0),
        "mean_throughput_tok_per_s": pytest.approx(1.0),
    }


def test_batch_of_two():
    b = SimContinuousBatcher(2, 10, 1)
    b.submit(0.0, 3)
    b.submit(0.0, 2)
    assert b.step(1) == []
    assert b.step(2) == [1]
    assert b.step(3) == [0]
    stats = b.stats()
    assert stats["mean_ttft"] == pytest.approx(1.0)
    assert stats["mean_tbt"] == pytest.approx(1.0)
    assert stats["mean_throughput_tok_per_s"] == pytest.approx(1.0)


def test_queueing_when_batch_full():
    b = SimContinuousBatcher(1, 10, 1)
    b.submit(0.0, 3)
    b.submit(0.0, 2)
    b.step(1)
    b.step(2)
    assert b.step(3) == [0]
    b.step(4)
    assert b.step(5) == [1]
    stats = b.stats()
    assert stats["mean_ttft"] == pytest.approx(2.5)
    assert stats["mean_tbt"] == pytest.approx(1.0)
    assert stats["mean_throughput_tok_per_s"] == pytest.approx(0.7)


def test_late_arrival_waits_for_its_step():
    b = SimContinuousBatcher(2, 10, 1)
    rid = b.submit(2.0, 2)
    b.step(1)
    b.step(2)
    assert b.step(3) == [rid]


def test_stats_empty_run_is_all_zero():
    b = SimContinuousBatcher(2, 10, 1)
    b.submit(0.0, 3)
    b.step(1)
    assert b.stats() == {"mean_ttft": 0.0, "mean_tbt": 0.0, "mean_throughput_tok_per_s": 0.0}


def test_gen_len_beyond_max_seq_len_raises():
    b = SimContinuousBatcher(2, 3, 1)
    with pytest.raises(ValueError):
        b.submit(0.0, 4)


def test_step_times_must_increase():
    b = SimContinuousBatcher(2, 10, 1)
    b.submit(0.0, 3)
    b.step(1)
    with pytest.raises(ValueError):
        b.step(1)


def test_invalid_constructor_args_raise():
    with pytest.raises(ValueError):
        SimContinuousBatcher(0, 10, 1)
    with pytest.raises(ValueError):
        SimContinuousBatcher(2, 0, 1)
    with pytest.raises(ValueError):
        SimContinuousBatcher(2, 10, 0)