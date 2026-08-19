# Day 3 — Metrics: TTFT, TPS, ITL; percentiles vs means

## Goal

Simulate streamed responses with realistic jitter, compute TTFT, ITL,
and TPS with the shared metric helpers, and see why the mean lies about
latency.

## Prerequisites

Python 3.10 or newer. The course environment with numpy and torch.
No downloads and no network access are needed.

## Concept

Three metrics carry most inference conversations:

- TTFT: time to first token. Gated by prefill, which is compute-bound.
  A streaming chat UI lives and dies on it.
- ITL: inter-token latency, the gap between consecutive streamed
  tokens. 10 ms ITL equals 100 tokens per second per user. Reading
  speed is about 10 tokens per second, so anything past that feels
  instant to a human.
- TPS: tokens per second. The name is ambiguous. Perceived TPS is one
  user's stream speed; total TPS is the system's aggregate. Say which
  one you mean.

Latency is right-skewed: a few slow requests drag the mean somewhere no
real request lives. Report P50, P90, and P99 instead. Performance work
pulls in the tail, not the mean.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson simulates 64 streamed responses (512 in full mode) with a
seeded random generator. Each response draws a prefill time and
per-token delays from a lognormal distribution, then builds the token
arrival timestamps. It computes TTFT per request, mean ITL via
`shared.metrics.inter_token_latency`, TPS via
`shared.metrics.tokens_per_sec`, and p50/p90/p95/p99 of TTFT via
`shared.metrics.percentile`.

## Expected observations

- The mean TTFT sits above the p50 TTFT. The skew is the whole point.
- Mean ITL of about 10 ms implies about 100 tokens per second per
  user.
- p95 minus p50 is a large fraction of p50, which is the tail that
  users actually feel.

## Metric

`ttft_p95_ms`: the 95th percentile of time-to-first-token in
milliseconds, with the mean, p50, and p90 printed alongside.

## Sources

- Virk, Inference Engineering, chapter "The metrics":
  https://www.arjunvirk.com/inference-engineering.html

## Hardware notes

Runs on CPU in both modes; the simulation is pure arithmetic, so no GPU
path exists. Full mode only increases the sample count.

## Reflection prompts

1. Why is the mean TTFT higher than the p50 TTFT?
2. For a blocking agent tool call with no streaming, which metric
   matters and which one is irrelevant?
3. What does a p95-minus-p50 gap of 200 ms tell you about your users?
4. When would you quote p99 instead of p95?