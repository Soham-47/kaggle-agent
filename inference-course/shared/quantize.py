"""Minimal asymmetric per-tensor quantization with torch tensors.

quantize() maps a float tensor into bits-wide integer levels using one
scale and one zero point for the whole tensor; dequantize() reverses it;
mse_error() scores the reconstruction. Constant tensors quantize to a
single level with scale 1.0.
"""

import torch


def quantize(tensor: torch.Tensor, bits: int):
    """Return (q, scale, zero) for tensor quantized to bits levels.

    q holds integer levels in [0, 2**bits - 1]; scale and zero are Python
    floats such that dequantize(q, scale, zero) approximates tensor.
    """
    if bits < 1:
        raise ValueError("bits must be at least 1")
    t = tensor.float()
    tmin = float(t.min())
    tmax = float(t.max())
    scale = (tmax - tmin) / (2**bits - 1)
    if scale == 0.0:
        scale = 1.0
    zero = tmin
    q = torch.clamp(torch.round((t - zero) / scale), 0, 2**bits - 1).to(torch.int64)
    return q, scale, zero


def dequantize(q: torch.Tensor, scale: float, zero: float) -> torch.Tensor:
    """Return the float approximation of q using the given scale and zero."""
    return zero + q.float() * scale


def mse_error(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """Return the mean squared error between two float tensors."""
    return float(torch.mean((orig - recon) ** 2))