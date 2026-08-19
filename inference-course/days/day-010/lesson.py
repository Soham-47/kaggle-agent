"""Day 10: arithmetic intensity; why decode is memory-bound.

Computes intensity for a 7B FP16 model at several sequence lengths,
classifies against the ridge, and in full mode on CUDA confirms the
memory-bound signature with a real matmul at batch 1 versus 128.
"""

import argparse

import numpy as np

from shared.env import device, has_cuda

PARAMS = 7e9
DTYPE_BYTES = 2      # FP16
RIDGE = 295.2        # H100 FP16 ops per byte
SEQ_LENGTHS = [1, 32, 256, 2048]


def intensity(seq_len: int) -> float:
    """FLOP per byte for one forward pass over seq_len tokens."""
    flops = 2.0 * PARAMS * seq_len
    bytes_moved = PARAMS * DTYPE_BYTES
    return flops / bytes_moved


def classify(i: float) -> str:
    return "memory-bound" if i < RIDGE else "compute-bound"


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 10 lesson: arithmetic intensity")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"

    print(f"mode={mode} params={PARAMS:.0f} dtype_bytes={DTYPE_BYTES} ridge={RIDGE}")
    for s in SEQ_LENGTHS:
        i = intensity(s)
        print(f"seq_len={s:>4} intensity_flop_per_byte={i:>8.1f} bound={classify(i)}")

    decode_i = intensity(1)
    prefill_i = intensity(2048)
    print(f"decode_intensity_flop_per_byte={decode_i:.1f}")
    print(f"prefill_intensity_at_2048={prefill_i:.0f}")

    if not args.smoke and has_cuda():
        import time

        import torch

        dim = 4096
        weight = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
        times = {}
        for batch in (1, 128):
            x = torch.randn(batch, dim, device="cuda", dtype=torch.float16)
            x @ weight  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x @ weight
            torch.cuda.synchronize()
            times[batch] = time.perf_counter() - t0
        ratio = times[128] / times[1]
        print(f"matmul_b1_ms={times[1] * 1000.0:.3f}")
        print(f"matmul_b128_ms={times[128] * 1000.0:.3f}")
        print(f"flops_ratio_b128_over_b1=128 time_ratio={ratio:.2f}")
    else:
        print("gpu_measure=skipped_no_cuda")


if __name__ == "__main__":
    main()