# Day 1 — What inference is; the runtime/infrastructure/tooling layers

## Goal

Time a real tiny model forward pass alone and under a burst of requests,
then classify the result into one of the three inference layers.

## Prerequisites

Python 3.10 or newer. The course environment with numpy and torch.
No downloads and no network access are needed.

## Concept

Training learns the weights. Inference runs the forward pass over those
weights to serve a product. A trained model is a file of numbers; it does
nothing until a runtime executes it.

Inference engineering is three stacked layers:

- Runtime: one model on one instance, as fast as physics allows.
  Batching, KV caches, quantization, and kernels live here.
- Infrastructure: many boxes, clusters, regions, and clouds, with high
  uptime. Autoscaling and load balancing live here.
- Tooling: the abstraction the builders use. Too black-box and you cannot
  tune anything. Too raw and nobody ships.

Diagnose the layer before you optimize. A slow single request is a runtime
problem. A service that is fast when idle and slow under load is an
infrastructure problem. A team that cannot express its needs has a tooling
problem.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson builds a three-layer MLP, warms it up, times one forward pass
20 times (100 times in full mode), then times eight back-to-back requests
on the same model. It reports mean and p95 latency through
`shared.metrics.percentile`.

## Expected observations

- The p95 latency is higher than the mean. Latency samples are
  right-skewed.
- The under-load mean is in the same order as the single-request
  mean, because the eight requests run back to back on the same
  model rather than truly in parallel. A real queueing effect needs
  concurrent requests on a busy engine; the point of the numbers is
  that one figure does not describe a service.
- In full mode on a CUDA machine the same model runs on the GPU and
  the latencies drop.

## Metric

`forward_latency_p95_ms`: the 95th percentile of single forward-pass
latency in milliseconds.

## Sources

- Virk, Inference Engineering, chapter "What inference even is":
  https://www.arjunvirk.com/inference-engineering.html

## Hardware notes

Smoke mode runs on CPU. Full mode moves the same model to CUDA when
available and calls `torch.cuda.synchronize()` so the timings are real.
No model downloads happen in either mode.

## Reflection prompts

1. A single request is slow while the GPU is idle. Which layer do you
   investigate first?
2. Why does a mean hide the p95 of a latency sample?
3. Name one runtime, one infrastructure, and one tooling decision a
   serving team makes.
4. When is the correct answer "call a hosted API and do no inference
   engineering at all"?