"""Day 2: latency, throughput, quality tradeoffs.

Times a real linear layer at several batch sizes and reports per-request
latency and request throughput. Batching trades per-user latency for
system throughput; quality is held constant and left to evals (day 5).
"""

import argparse
import time

import torch

from shared.env import device, device_name, has_cuda
from shared.metrics import throughput_per_req


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 2 lesson: latency vs throughput")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    dev = "cpu" if args.smoke else device()
    dim = 256 if args.smoke else 1024
    batches = [1, 4, 16] if args.smoke else [1, 8, 32, 64]
    reps = 20 if args.smoke else 50

    torch.manual_seed(0)
    weight = torch.randn(dim, dim, device=dev)

    label = f" ({device_name()})" if dev == "cuda" else ""
    print(f"mode={mode} device={dev}{label} dim={dim}")

    results = {}
    for batch in batches:
        x = torch.randn(batch, dim, device=dev)
        x @ weight  # warmup
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            x @ weight
        if dev == "cuda":
            torch.cuda.synchronize()
        elapsed_s = (time.perf_counter() - t0) / reps
        latency_ms = elapsed_s * 1000.0
        req_per_s = throughput_per_req([elapsed_s] * batch)
        results[batch] = (latency_ms, req_per_s)
        print(f"batch={batch} latency_ms={latency_ms:.3f} throughput_req_per_s={req_per_s:.1f}")

    print(f"latency_ms_batch1={results[1][0]:.3f}")
    largest = batches[-1]
    print(f"throughput_req_per_s={results[largest][1]:.1f}")


if __name__ == "__main__":
    main()