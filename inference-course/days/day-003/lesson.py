"""Day 3: metrics TTFT, TPS, ITL; percentiles vs means.

Simulates streamed responses with seeded lognormal jitter and reports
TTFT percentiles, mean ITL, and mean TPS. The mean-vs-p50 gap shows why
latency reporting uses percentiles.
"""

import argparse
import math

import numpy as np

from shared.metrics import inter_token_latency, percentile, tokens_per_sec


def simulate_stream(rng: np.random.Generator, n_tokens: int) -> tuple[float, list[float]]:
    """Return (ttft_s, token arrival timestamps in seconds)."""
    prefill_s = float(rng.lognormal(mean=math.log(0.150), sigma=0.35))
    delays = rng.lognormal(mean=math.log(0.010), sigma=0.30, size=n_tokens)
    stamps = [prefill_s]
    for d in delays:
        stamps.append(stamps[-1] + d)
    return prefill_s, stamps[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 3 lesson: TTFT, ITL, TPS")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    n_requests = 64 if args.smoke else 512
    n_tokens = 40 if args.smoke else 120

    rng = np.random.default_rng(42)
    ttfts = []
    itls = []
    tpss = []
    for _ in range(n_requests):
        ttft_s, stamps = simulate_stream(rng, n_tokens)
        ttfts.append(ttft_s * 1000.0)
        itls.append(inter_token_latency(stamps) * 1000.0)
        tpss.append(tokens_per_sec(n_tokens, stamps[-1]))

    mean_ttft = sum(ttfts) / len(ttfts)
    print(f"mode={mode} requests={n_requests} tokens_per_request={n_tokens}")
    print(f"ttft_mean_ms={mean_ttft:.1f}")
    print(f"ttft_p50_ms={percentile(ttfts, 50):.1f}")
    print(f"ttft_p90_ms={percentile(ttfts, 90):.1f}")
    print(f"ttft_p95_ms={percentile(ttfts, 95):.1f}")
    print(f"ttft_p99_ms={percentile(ttfts, 99):.1f}")
    print(f"itl_mean_ms={sum(itls) / len(itls):.1f}")
    print(f"tps_mean={sum(tpss) / len(tpss):.1f}")


if __name__ == "__main__":
    main()