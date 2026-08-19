import pytest

from shared import metrics


def test_percentile_median():
    assert metrics.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)


def test_percentile_min_and_max():
    assert metrics.percentile([1, 2, 3, 4], 0) == 1.0
    assert metrics.percentile([1, 2, 3, 4], 100) == 4.0


def test_percentile_single_value():
    assert metrics.percentile([7], 90) == 7.0


def test_percentile_unsorted_input():
    assert metrics.percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        metrics.percentile([], 50)


def test_percentile_out_of_range_p_raises():
    with pytest.raises(ValueError):
        metrics.percentile([1, 2], -1)
    with pytest.raises(ValueError):
        metrics.percentile([1, 2], 101)


def test_inter_token_latency_mean_of_gaps():
    assert metrics.inter_token_latency([0.0, 0.2, 0.5]) == pytest.approx(0.25)


def test_inter_token_latency_short_input_is_zero():
    assert metrics.inter_token_latency([1.0]) == 0.0
    assert metrics.inter_token_latency([]) == 0.0


def test_tokens_per_sec():
    assert metrics.tokens_per_sec(100, 10) == pytest.approx(10.0)


def test_tokens_per_sec_zero_elapsed_raises():
    with pytest.raises(ValueError):
        metrics.tokens_per_sec(100, 0)


def test_throughput_per_req():
    assert metrics.throughput_per_req([0.5, 0.5]) == pytest.approx(2.0)


def test_throughput_per_req_empty_is_zero():
    assert metrics.throughput_per_req([]) == 0.0


def test_throughput_per_req_zero_times_raise():
    with pytest.raises(ValueError):
        metrics.throughput_per_req([0.0, 1.0])
