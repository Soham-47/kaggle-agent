"""Latency and throughput metrics for inference experiments.

Pure-Python helpers; lessons use them to report TTFT, inter-token
latency, tokens per second, and request throughput.
"""


def percentile(values, p: float) -> float:
    """Return the p-th percentile of values with linear interpolation.

    p must be between 0 and 100. An empty sequence raises ValueError.
    """
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be between 0 and 100")
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    rank = p / 100.0 * (len(data) - 1)
    lo = min(int(rank), len(data) - 2)
    frac = rank - lo
    return data[lo] * (1 - frac) + data[lo + 1] * frac


def inter_token_latency(stream_times_s) -> float:
    """Return the mean time in seconds between consecutive streamed tokens.

    stream_times_s holds the timestamp of each token as it arrived.
    Fewer than two timestamps yield 0.0.
    """
    if len(stream_times_s) < 2:
        return 0.0
    deltas = [b - a for a, b in zip(stream_times_s, stream_times_s[1:])]
    return sum(deltas) / len(deltas)


def tokens_per_sec(total_tokens: int, elapsed_s: float) -> float:
    """Return tokens generated per second; elapsed_s must be positive."""
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive")
    return total_tokens / elapsed_s


def throughput_per_req(req_times_s) -> float:
    """Return requests completed per second from per-request durations.

    Each entry in req_times_s is one request's duration in seconds.
    An empty list yields 0.0.
    """
    if not req_times_s:
        return 0.0
    if any(t <= 0 for t in req_times_s):
        raise ValueError("request times must be positive")
    return len(req_times_s) / sum(req_times_s)
