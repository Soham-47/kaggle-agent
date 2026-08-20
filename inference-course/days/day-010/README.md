# Day 10 — Arithmetic intensity; why decode is memory-bound

## Goal

Compute the arithmetic intensity of prefill and decode for a real model
size, place both against the ridge point, and in full mode confirm the
memory-bound signature with a real matmul at batch sizes 1 and 128.

## Prerequisites

Python 3.10 or newer. The course environment with numpy and torch.
No downloads and no network access are needed.

## Concept

Arithmetic intensity is the matching property of an algorithm: total
compute work divided by total memory traffic. The roofline ridge is the
device's own ops-per-byte ratio. Compare the two and you know the
bottleneck.

Decode generates one token per forward pass. It does about 2P
operations for a model with P parameters and reads about 2P bytes of
FP16 weights, so its intensity is about 1 operation per byte: far below
the H100 ridge of about 295. Decode is memory-bound. It spends its
life waiting on HBM while the tensor cores sit idle, and it sets TPS.

Prefill processes S tokens per pass. It does about 2PS operations for
the same 2P bytes of weights, so its intensity is about S operations
per byte: compute-bound for long prompts, and it sets TTFT.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson computes intensity for a 7-billion-parameter model in FP16
at sequence lengths 1, 32, 256, and 2048, classifies each against the
ridge, and prints the decode and prefill intensities. Full mode on
CUDA times a real `(B, 4096) @ (4096, 4096)` FP16 matmul at batch 1
(decode-like) and batch 128 (prefill-like) and reports the wall times
and the flop ratio.

## Expected observations

- Intensity equals the sequence length: 1 for decode, 2048 for a
  2048-token prefill.
- Decode sits far left of the ridge; prefill at 2048 sits far right.
- On GPU, batch 128 does 128 times the arithmetic of batch 1 but
  takes only a small multiple of the time, because both read the
  same weight bytes. That flat scaling is the memory-bound
  signature.

## Metric

`decode_intensity_flop_per_byte`: the arithmetic intensity of one
decode step, with `prefill_intensity_at_2048` and the classification
table printed alongside.

## Sources

- Virk, Inference Engineering, chapter "The bottleneck math":
  https://www.arjunvirk.com/inference-engineering.html

## Hardware notes

Smoke mode runs on CPU and does the arithmetic only. Full mode without
CUDA prints the same table and skips the timing. Full mode with CUDA
times real matmuls with `torch.cuda.synchronize()`.

## Reflection prompts

1. Why does decode reload the entire weight matrix for a single
   token?
2. Batching raises decode's intensity. Why, and what sets the limit?
3. Why is buying a faster GPU of limited use for decode-bound
   traffic?
4. Which metric does the memory-bound phase set: TTFT or TPS?