from shared import env


def test_has_cuda_returns_bool():
    assert isinstance(env.has_cuda(), bool)


def test_device_is_cpu_or_cuda():
    assert env.device() in ("cpu", "cuda")


def test_device_matches_cuda_availability():
    assert (env.device() == "cuda") == env.has_cuda()


def test_device_name_nonempty():
    assert isinstance(env.device_name(), str)
    assert env.device_name()


def test_vram_available_gb_matches_cuda():
    vram = env.vram_available_gb()
    if env.has_cuda():
        assert vram is not None
        assert vram > 0
    else:
        assert vram is None