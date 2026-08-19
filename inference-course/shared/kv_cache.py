"""Simulated KV cache allocators: contiguous and paged.

Both classes model KV memory in fp32 bytes: each token stores K and V
for every layer, so bytes_per_token is
num_layers * num_heads * head_dim * 2 (K plus V) * 4 bytes.
"""


class ContiguousKVAllocator:
    """A contiguous KV cache: one flat buffer with a fixed token capacity.

    allocate() reserves tokens from the buffer; use() marks tokens that
    actually store K/V data; free() releases reservations. used_tokens
    never exceeds allocated_tokens. A cache at usage_ratio() == 1.0 is
    full of live data.
    """

    def __init__(self, max_tokens: int, num_layers: int, num_heads: int, head_dim: int):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.max_tokens = max_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.allocated_tokens = 0
        self.used_tokens = 0

    @property
    def bytes_per_token(self) -> float:
        """Return the fp32 bytes one token of K/V needs across all layers."""
        return self.num_layers * self.num_heads * self.head_dim * 2 * 4

    def allocate(self, num_tokens: int) -> None:
        """Reserve num_tokens of contiguous space; raise MemoryError when full."""
        if num_tokens < 1:
            raise ValueError("num_tokens must be at least 1")
        if self.allocated_tokens + num_tokens > self.max_tokens:
            raise MemoryError("contiguous KV cache has no room for the allocation")
        self.allocated_tokens += num_tokens

    def use(self, num_tokens: int) -> None:
        """Record that num_tokens of K/V were actually written; clamps to allocated."""
        if num_tokens < 0:
            raise ValueError("num_tokens must not be negative")
        self.used_tokens = min(self.used_tokens + num_tokens, self.allocated_tokens)

    def free(self, num_tokens: int) -> None:
        """Release a reservation of num_tokens; floors at zero."""
        if num_tokens < 0:
            raise ValueError("num_tokens must not be negative")
        self.allocated_tokens = max(0, self.allocated_tokens - num_tokens)
        self.used_tokens = max(0, self.used_tokens - num_tokens)

    def usage_ratio(self) -> float:
        """Return the fraction of capacity that stores live K/V data."""
        return self.used_tokens / self.max_tokens


class PagedKVAllocator:
    """A paged KV cache: num_blocks blocks of block_size tokens each.

    Allocations round up to whole blocks, so a request's last block can
    be partially used; allocated_tokens minus used_tokens is that
    block-level slack (internal fragmentation).
    """

    def __init__(self, block_size: int, num_blocks: int, num_layers: int, num_heads: int, head_dim: int):
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.capacity_tokens = block_size * num_blocks
        self.allocated_tokens = 0
        self.used_tokens = 0

    @property
    def bytes_per_token(self) -> float:
        """Return the fp32 bytes one token of K/V needs across all layers."""
        return self.num_layers * self.num_heads * self.head_dim * 2 * 4

    def allocate(self, num_tokens: int) -> None:
        """Reserve space for num_tokens, rounded up to whole blocks."""
        if num_tokens < 1:
            raise ValueError("num_tokens must be at least 1")
        blocks = (num_tokens + self.block_size - 1) // self.block_size
        if self.allocated_tokens + blocks * self.block_size > self.capacity_tokens:
            raise MemoryError("paged KV cache has no room for the allocation")
        self.allocated_tokens += blocks * self.block_size

    def use(self, num_tokens: int) -> None:
        """Record that num_tokens of K/V were actually written; clamps to allocated."""
        if num_tokens < 0:
            raise ValueError("num_tokens must not be negative")
        self.used_tokens = min(self.used_tokens + num_tokens, self.allocated_tokens)

    def free(self, num_tokens: int) -> None:
        """Release a reservation of num_tokens, rounded up to whole blocks."""
        if num_tokens < 0:
            raise ValueError("num_tokens must not be negative")
        blocks = (num_tokens + self.block_size - 1) // self.block_size
        self.allocated_tokens = max(0, self.allocated_tokens - blocks * self.block_size)
        self.used_tokens = max(0, self.used_tokens - num_tokens)

    def usage_ratio(self) -> float:
        """Return the fraction of capacity that stores live K/V data."""
        return self.used_tokens / self.capacity_tokens
