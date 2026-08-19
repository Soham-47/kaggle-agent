"""Day 5: model selection and evals; shared vs dedicated.

Runs a seeded Monte Carlo eval over two candidate models, measures
accuracy and p95 latency, then prices the same traffic on a shared
per-token API versus dedicated GPUs.
"""

import argparse
import math

import numpy as np

from shared.metrics import percentile

REQ_PER_S = 5.0
TOKENS_PER_REQ = 1200
SHARED_PRICE_PER_M = 1.50
HOURS_PER_DAY = 24.0
SECONDS_PER_DAY = 86400.0
CANDIDATES = {
    "a": {"true_acc": 0.86, "mean_latency_ms": 80.0, "gpu_usd_per_hour": 2.20},
    "b": {"true_acc": 0.93, "mean_latency_ms": 140.0, "gpu_usd_per_hour": 3.90},
}


def run_eval(rng: np.random.Generator, n_items: int, true_acc: float, mean_ms: float) -> tuple[float, float]:
    """Return (measured accuracy, p95 latency in ms) for one candidate."""
    correct = rng.random(n_items) < true_acc
    latencies = rng.lognormal(mean=math.log(mean_ms), sigma=0.30, size=n_items)
    return float(correct.mean()), percentile(latencies.tolist(), 95)


def shared_cost_per_day() -> float:
    tokens_per_day = REQ_PER_S * SECONDS_PER_DAY * TOKENS_PER_REQ
    return tokens_per_day / 1e6 * SHARED_PRICE_PER_M


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 5 lesson: model selection and evals")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    n_items = 80 if args.smoke else 200

    results = {}
    for name, spec in CANDIDATES.items():
        seed = 1 if name == "a" else 2
        acc, p95 = run_eval(np.random.default_rng(seed), n_items, spec["true_acc"], spec["mean_latency_ms"])
        results[name] = (acc, p95)
        print(f"model={name} eval_accuracy={acc:.3f} latency_p95_ms={p95:.1f}")

    shared = shared_cost_per_day()
    print(f"shared_cost_per_day_usd={shared:.2f}")
    for name, spec in CANDIDATES.items():
        dedicated = spec["gpu_usd_per_hour"] * HOURS_PER_DAY
        print(f"model={name} dedicated_cost_per_day_usd={dedicated:.2f}")

    print(f"eval_accuracy_a={results['a'][0]:.3f}")
    print(f"eval_accuracy_b={results['b'][0]:.3f}")


if __name__ == "__main__":
    main()