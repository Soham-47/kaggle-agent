"""Day 7: prefill vs decode; the autoregressive loop.

Builds a tiny GRU language model. Prefill processes the whole context
in one call; decode runs one call per generated token. Times both and
prints the call counts and per-token decode latency.
"""

import argparse
import time

import torch

from shared.env import device, device_name, has_cuda

VOCAB = 16
EMBED = 16
HIDDEN = 64


class TinyLM(torch.nn.Module):
    """Embedding + GRU (prefill) + GRUCell (decode) + linear head."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB, EMBED)
        self.gru = torch.nn.GRU(EMBED, HIDDEN, batch_first=True)
        self.cell = torch.nn.GRUCell(EMBED, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB)
        with torch.no_grad():
            self.cell.weight_ih.copy_(self.gru.weight_ih_l0)
            self.cell.weight_hh.copy_(self.gru.weight_hh_l0)
            self.cell.bias_ih.copy_(self.gru.bias_ih_l0)
            self.cell.bias_hh.copy_(self.gru.bias_hh_l0)

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        """Process the whole context in one call; return the final state."""
        out, _ = self.gru(self.emb(ids).unsqueeze(0))
        return out[:, -1, :]

    def step(self, token_id: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one decode step; return (logits, next state)."""
        h = self.cell(self.emb(token_id), h)
        return self.head(h), h


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 7 lesson: prefill vs decode")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    dev = "cpu" if args.smoke else device()
    ctx_len = 64 if args.smoke else 256
    gen_len = 32 if args.smoke else 128

    torch.manual_seed(0)
    model = TinyLM().to(dev)
    ctx = torch.arange(VOCAB).repeat((ctx_len + VOCAB - 1) // VOCAB)[:ctx_len].to(dev)

    model.prefill(ctx)  # warmup
    sync(dev)
    t0 = time.perf_counter()
    h = model.prefill(ctx)
    sync(dev)
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    steps_ms = []
    token = ctx[-1].unsqueeze(0)
    for _ in range(gen_len):
        sync(dev)
        t0 = time.perf_counter()
        logits, h = model.step(token, h)
        sync(dev)
        steps_ms.append((time.perf_counter() - t0) * 1000.0)
        token = logits.argmax(dim=-1)

    label = f" ({device_name()})" if dev == "cuda" else ""
    print(f"mode={mode} device={dev}{label}")
    print(f"prefill_ms={prefill_ms:.2f}")
    print(f"decode_ms_per_token={sum(steps_ms) / len(steps_ms):.3f}")
    print(f"decode_total_ms={sum(steps_ms):.2f}")
    print(f"forward_calls_prefill=1 forward_calls_decode={gen_len}")


if __name__ == "__main__":
    main()