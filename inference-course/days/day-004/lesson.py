"""Day 4: requirements, interface, latency budget, unit economics, usage.

A deterministic requirements calculator for a fixed product spec. Prints
tokens per day, cost per day, required decode TPS at peak, TTFT and ITL
budget checks, and margin per request.
"""

import argparse

REQ_PER_S_STEADY = 10.0
SPIKE_MULTIPLIER = 3.0
INPUT_TOKENS = 1500
OUTPUT_TOKENS = 300
PRICE_PER_M = 1.20
TTFT_BUDGET_MS = 400.0
ITL_BUDGET_MS = 20.0
PREFILL_TPS = 8000.0
ENGINE_ITL_MS = 12.0
REVENUE_PER_REQ = 0.002
SECONDS_PER_DAY = 86400.0


def calculate() -> dict:
    tokens_per_req = INPUT_TOKENS + OUTPUT_TOKENS
    req_per_s_peak = REQ_PER_S_STEADY * SPIKE_MULTIPLIER
    tokens_per_day = tokens_per_req * REQ_PER_S_STEADY * SECONDS_PER_DAY
    cost_per_day = tokens_per_day / 1e6 * PRICE_PER_M
    required_decode_tps_steady = REQ_PER_S_STEADY * OUTPUT_TOKENS
    required_decode_tps_peak = req_per_s_peak * OUTPUT_TOKENS
    ttft_ms = INPUT_TOKENS / PREFILL_TPS * 1000.0
    margin_per_req = REVENUE_PER_REQ - tokens_per_req / 1e6 * PRICE_PER_M
    return {
        "tokens_per_req": tokens_per_req,
        "req_per_s_peak": req_per_s_peak,
        "tokens_per_day": tokens_per_day,
        "cost_per_day": cost_per_day,
        "required_decode_tps_steady": required_decode_tps_steady,
        "required_decode_tps_peak": required_decode_tps_peak,
        "ttft_ms": ttft_ms,
        "ttft_budget_ok": ttft_ms <= TTFT_BUDGET_MS,
        "itl_budget_ok": ENGINE_ITL_MS <= ITL_BUDGET_MS,
        "margin_per_req": margin_per_req,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 4 lesson: requirements calculator")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"

    r = calculate()
    print(f"mode={mode} spec='10 req/s steady, 3x spike, 1500+300 tokens, $1.20/M'")
    print(f"tokens_per_req={r['tokens_per_req']}")
    print(f"tokens_per_day={r['tokens_per_day']:.0f}")
    print(f"cost_per_day_usd={r['cost_per_day']:.2f}")
    print(f"required_decode_tps_steady={r['required_decode_tps_steady']:.0f}")
    print(f"required_decode_tps_peak={r['required_decode_tps_peak']:.0f}")
    print(f"ttft_expected_ms={r['ttft_ms']:.0f} budget_ms={TTFT_BUDGET_MS:.0f} ok={r['ttft_budget_ok']}")
    print(f"itl_expected_ms={ENGINE_ITL_MS:.0f} budget_ms={ITL_BUDGET_MS:.0f} ok={r['itl_budget_ok']}")
    print(f"margin_per_req_usd={r['margin_per_req']:.4f}")

    if not args.smoke:
        alt = dict(r)
        alt["cost_per_day"] = r["tokens_per_day"] / 1e6 * 4.00
        print(f"premium_engine_cost_per_day_usd={alt['cost_per_day']:.2f}")


if __name__ == "__main__":
    main()