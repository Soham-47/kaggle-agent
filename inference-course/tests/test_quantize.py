import pytest
import torch

from shared.quantize import dequantize, mse_error, quantize


def test_quantize_levels_in_range():
    t = torch.arange(8, dtype=torch.float32)
    q, scale, zero = quantize(t, 8)
    assert not q.is_floating_point()
    assert q.min() >= 0
    assert q.max() <= 255


def test_roundtrip_endpoints_exact():
    t = torch.tensor([0.0, 3.0], dtype=torch.float32)
    q, scale, zero = quantize(t, 8)
    recon = dequantize(q, scale, zero)
    assert recon.min() == pytest.approx(0.0)
    assert recon.max() == pytest.approx(3.0)


def test_more_bits_means_less_error():
    t = torch.linspace(-1.0, 1.0, 1000)
    q8, s8, z8 = quantize(t, 8)
    q4, s4, z4 = quantize(t, 4)
    err8 = mse_error(t, dequantize(q8, s8, z8))
    err4 = mse_error(t, dequantize(q4, s4, z4))
    assert err8 < err4


def test_mse_error_perfect_reconstruction_is_zero():
    t = torch.randn(64)
    assert mse_error(t, t) == 0.0


def test_mse_error_positive_for_mismatch():
    t = torch.zeros(4)
    assert mse_error(t, torch.ones(4)) == pytest.approx(1.0)


def test_constant_tensor_quantizes():
    t = torch.full((8,), 2.0)
    q, scale, zero = quantize(t, 8)
    recon = dequantize(q, scale, zero)
    assert (recon == 2.0).all()


def test_invalid_bits_raise():
    with pytest.raises(ValueError):
        quantize(torch.zeros(4), 0)