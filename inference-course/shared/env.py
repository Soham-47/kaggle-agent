"""Environment helpers: CUDA detection, device selection, VRAM queries.

Every module that needs the GPU asks these helpers first. CUDA-only code
paths in lessons and tests must be guarded with has_cuda() and marked
@pytest.mark.gpu in tests.
"""


def has_cuda() -> bool:
    """Return True when a CUDA-capable device is available to PyTorch."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def device_name() -> str:
    """Return the name of the active device; "cpu" when no GPU is present."""
    if has_cuda():
        import torch

        return torch.cuda.get_device_name(0)
    return "cpu"


def device() -> str:
    """Return "cuda" when a CUDA device is available, else "cpu"."""
    return "cuda" if has_cuda() else "cpu"


def vram_available_gb() -> float | None:
    """Return free VRAM on device 0 in GiB; return None on CPU-only hosts."""
    if not has_cuda():
        return None
    import torch

    free_bytes, _ = torch.cuda.mem_get_info(0)
    return free_bytes / (1024**3)
