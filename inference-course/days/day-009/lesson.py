"""Day 9: the roofline model; compute-bound vs memory-bound.

Builds a roofline for an H100-style device, classifies an elementwise
add and a large matmul, and in full mode on CUDA confirms the
classification with real timings.
"""

import argparse

import numpy as np

from shared.env import device, has_cuda

PEAK_FLOPS = 989e12   # H100 FP16 tensor-core FLOP/s
PEAK_BW = 3.35e12     # H100 HBM bytes/s
INTENSITIES = [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 10000.0]


def attainable(peak_flops: float, peak_bw: float, intensities) -> np.ndarray:
    """Return the attainable FLOP/s for each intensity under the roofline."""
    return np.minimum(peak_flops, np.asarray(intensities, dtype=float) * peak_bw)


def classify(intensity: float, ridge: float) -> str:
    return "memory-bound" if intensity < ridge else "compute-bound"


def op_elementwise_add(n: int) -> float:
    """FLOP per byte for adding two n-vectors in FP32."""
    flops = n
    bytes_moved = 3 * n * 4
    return flops / bytes_moved


def op_matmul(n: int) -> float:
    """FLOP per byte for an n x n FP16 matmul."""
    flops = 2 * n**3
    bytes_moved = 3 * n * n * 2
    return flops / bytes_moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 9 lesson: the roofline model")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"

    ridge = PEAK_FLOPS / PEAK_BW
    vals = attainable(PEAK_FLOPS, PEAK_BW, INTENSITIES)
    print(f"mode={mode} peak_flops={PEAK_FLOPS / 1e12:.0f}tflops peak_bw={PEAK_BW / 1e12:.2f}tb_per_s")
    print(f"ridge_intensity_ops_per_byte={ridge:.1f}")
    for i, v in zip(INTENSITIES, vals):
        print(f"intensity={i:>8.1f} attainable_tflops={v / 1e12:>9.2f} bound={classify(i, ridge)}")

    add_i = op_elementwise_add(1_000_000)
    mm_i = op_matmul(4096)
    print(f"op=elementwise_add intensity={add_i:.3f} bound={classify(add_i, ridge)}")
    print(f"op=matmul_4096 intensity={mm_i:.1f} bound={classify(mm_i, ridge)}")

    if not args.smoke and has_cuda():
        import time

        import torch

        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        b = torch.empty_like(a)
        a @ a  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ a
        torch.cuda.synchronize()
        mm_s = time.perf_counter() - t0
        achieved_flops = 2 * 4096**3 / mm_s
        b.copy_(a)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        copy_s = time.perf_counter() - t0
        achieved_bw = 2 * 4096 * 4096 * 2 / copy_s
        print(f"measured_matmul_tflops={achieved_flops / 1e12:.0f} ({achieved_flops / PEAK_FLOPS * 100:.0f}% of ceiling)")
        print(f"measured_copy_gb_per_s={achieved_bw / 1e9:.0f} ({achieved_bw / PEAK_BW * 100:.0f}% of ceiling)")
    else:
        print("gpu_measure=skipped_no_cuda")


if __name__ == "__main__":
    main()