"""Day 1: what inference is; the runtime, infrastructure, tooling layers.

Times a tiny MLP forward pass alone (runtime work) and under a burst of
eight back-to-back requests (infrastructure-style load). Reports mean and
p95 latency for each.
"""

import argparse
import time

import torch

from shared.env import device, device_name, has_cuda
from shared.metrics import percentile


def build_model(dev: str) -> torch.nn.Module:
    """Return a three-layer MLP on dev with fixed seed weights."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
    )
    return model.to(dev)


def time_forward(model, x, n_runs: int, dev: str) -> list[float]:
    """Time n_runs forward passes; return latencies in milliseconds."""
    latencies_ms = []
    for _ in range(n_runs):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 1 lesson: inference layers")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    dev = "cpu" if args.smoke else device()
    n_runs = 20 if args.smoke else 100

    model = build_model(dev)
    x = torch.randn(1, 64, device=dev)
    model(x)  # warmup

    single = time_forward(model, x, n_runs, dev)
    under_load = time_forward(model, x, 8, dev)

    label = f" ({device_name()})" if dev == "cuda" else ""
    print(f"mode={mode} device={dev}{label}")
    print(f"forward_latency_mean_ms={sum(single) / len(single):.3f}")
    print(f"forward_latency_p95_ms={percentile(single, 95):.3f}")
    print(f"under_load_latency_mean_ms={sum(under_load) / len(under_load):.3f}")


if __name__ == "__main__":
    main()