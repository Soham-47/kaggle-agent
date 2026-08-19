# Day 7 — Prefill vs decode; the autoregressive loop

## Goal

Time the two phases of generation on a real tiny recurrent language
model: one prefill call over the whole context versus one decode call
per generated token.

## Prerequisites

Python 3.10 or newer. The course environment with numpy and torch.
No downloads and no network access are needed.

## Concept

Generation has two phases with different cost profiles.

Prefill processes the entire input prompt in one forward pass and
builds the KV cache. It is compute-bound, and it sets TTFT.

Decode is the autoregressive loop: one forward pass per output token,
each waiting on the last, each reading the weights and the growing KV
cache. It is memory-bound, and it sets TPS. The token at position N+1
cannot start until token N exists. That sequential dependency is the
reason a 500-token answer means 500 sequential forward passes, and it
is the thing that makes concurrent serving hard.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson builds a tiny language model: an embedding, a GRU that
processes the whole context in one call, a GRU cell that runs one
decode step at a time, and a linear head over a 16-token vocabulary.
It prefills a context of 64 tokens (256 in full mode) and greedily
generates 32 tokens (128 in full mode), timing the prefill call and
every decode step.

## Expected observations

- Prefill finishes the whole context in one call. Its wall time is the
  whole-context time.
- Decode needs exactly one call per generated token; the lesson prints
  the call counts side by side.
- Mean decode time per token is the number that matters for streaming:
  its inverse is the per-user tokens per second.

## Metric

`decode_ms_per_token`: the mean wall time of one decode step in
milliseconds, with `prefill_ms` and the forward call counts printed
alongside.

## Sources

- Virk, Inference Engineering, chapter "LLM mechanics":
  https://www.arjunvirk.com/inference-engineering.html
- Modular, LLM Inference Handbook, "Attention and the KV cache":
  https://handbook.modular.com/inference-optimization/pagedattention/

## Hardware notes

Smoke mode runs on CPU with a 64-token context. Full mode runs on CUDA
when available with a 256-token context and synchronizes timing with
`torch.cuda.synchronize()`. The GRU and the decode cell share one
parameter set, copied at construction.

## Reflection prompts

1. Why can the model not emit token N+1 before token N?
2. Which phase sets TTFT, and why?
3. Which phase sets TPS, and why?
4. Where does the KV cache get built, and where does it get read?