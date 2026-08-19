import pytest

from shared.kv_cache import ContiguousKVAllocator, PagedKVAllocator

LAYERS, HEADS, HEAD_DIM = 2, 4, 8


def test_contiguous_bytes_per_token():
    a = ContiguousKVAllocator(max_tokens=100, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    assert a.bytes_per_token == LAYERS * HEADS * HEAD_DIM * 2 * 4


def test_contiguous_allocate_use_ratio():
    a = ContiguousKVAllocator(max_tokens=100, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    a.allocate(30)
    a.use(30)
    assert a.allocated_tokens == 30
    assert a.used_tokens == 30
    assert a.usage_ratio() == pytest.approx(0.3)


def test_contiguous_overflow_raises():
    a = ContiguousKVAllocator(max_tokens=100, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    a.allocate(60)
    with pytest.raises(MemoryError):
        a.allocate(41)


def test_contiguous_free_releases():
    a = ContiguousKVAllocator(max_tokens=100, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    a.allocate(50)
    a.use(50)
    a.free(20)
    assert a.allocated_tokens == 30
    assert a.used_tokens == 30


def test_contiguous_use_clamps_to_allocated():
    a = ContiguousKVAllocator(max_tokens=100, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    a.allocate(10)
    a.use(25)
    assert a.used_tokens == 10


def test_paged_blocks_round_up():
    a = PagedKVAllocator(block_size=4, num_blocks=10, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    assert a.capacity_tokens == 40
    a.allocate(6)
    assert a.allocated_tokens == 8


def test_paged_partial_block_fragmentation():
    a = PagedKVAllocator(block_size=4, num_blocks=10, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    a.allocate(6)
    a.use(6)
    assert a.allocated_tokens == 8
    assert a.used_tokens == 6
    assert a.usage_ratio() == pytest.approx(6 / 40)


def test_paged_overflow_raises():
    a = PagedKVAllocator(block_size=4, num_blocks=10, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    a.allocate(36)
    with pytest.raises(MemoryError):
        a.allocate(5)


def test_paged_free_rounds_to_blocks():
    a = PagedKVAllocator(block_size=4, num_blocks=10, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    a.allocate(10)
    a.free(5)
    assert a.allocated_tokens == 4


def test_invalid_capacity_raises():
    with pytest.raises(ValueError):
        ContiguousKVAllocator(max_tokens=0, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    with pytest.raises(ValueError):
        PagedKVAllocator(block_size=0, num_blocks=10, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)
    with pytest.raises(ValueError):
        PagedKVAllocator(block_size=4, num_blocks=0, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)