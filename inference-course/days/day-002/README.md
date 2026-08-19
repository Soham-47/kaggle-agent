# Day 2 — Latency, throughput, quality tradeoffs

## Goal

Measure per-request latency and request throughput at several batch
sizes on a real linear layer, and read the tradeoff off the numbers.

## Prerequisites

Python 3.10 or newer. The course environment with numpy and torch.
No downloads and no network access are needed.

## Concept

Three quantities always trade against each other: latency (how fast one
user gets an answer), throughput (how many tokens or requests the whole
system pushes per second), and quality (how good the outputs are). You
cannot maximize all three.

Batching is the classic lever. A bigger batch lifts throughput, because
one weight load is amortized over more requests. The same bigger batch
hurts per-user latency, because a request waits for the batch to form
and the batch step takes longer. Quantization lifts both latency and
throughput but can dent quality. The job is to find the point on the
tradeoff surface your product actually needs, not to chase a benchmark.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson times a real `(batch, dim) @ (dim, dim)` matmul at batch
sizes 1, 4, and 16 (smoke mode; full mode adds 32 and 64 and uses a
larger dim). For each batch it reports the mean wall time per batch and
the request throughput via `shared.metrics.throughput_per_req`.

## Expected observations

- Request throughput grows with batch size.
- Per-request latency stays roughly flat at these tiny sizes, because
  even batch 1 already saturates the matmul's memory traffic. At real
  model sizes, per-request latency does rise with the batch; the
  trend to look for is throughput growing far faster than latency.
- The tradeoff is visible as a ratio: a 16x batch buys well short of a
  16x latency penalty, which is why serving engines batch.

## Metric

`throughput_req_per_s`: requests completed per second at the largest
tested batch size, with `latency_ms_batch1` printed alongside as the
contrast point.

## Sources

- Virk, Inference Engineering, chapter "Latency, throughput, quality":
  https://www.arjunvirk.com/inference-engineering.html

## Hardware notes

Smoke mode runs on CPU with dim 256. Full mode runs on CUDA when
available with dim 1024 and synchronizes timing with
`torch.cuda.synchronize()`. Quality is not measured here; evals are the
subject of day 5.

## Reflection prompts

1. Why does a batch of 16 requests finish in less than 16 times the
   single-request time?
2. What budget would make batching the wrong choice for a product?
3. Which side of the tradeoff does quantization move, and what does it
   risk?
4. How would you phrase a spec that pins one point on this tradeoff
   surface?