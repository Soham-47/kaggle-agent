# Day 9 — The roofline model; compute-bound vs memory-bound

## Goal

Build a roofline for a configurable device, classify two textbook
operations, and in full mode confirm the classification with real
timings on a CUDA GPU.

## Prerequisites

Python 3.10 or newer. The course environment with numpy and torch.
No downloads and no network access are needed.

## Concept

A GPU has exactly two resources you can run out of: compute (floating
point operations per second) and memory bandwidth (bytes per second
between HBM and the compute units). At any moment one is saturated and
the other sits idle. The idle one is your bottleneck.

The roofline model draws both ceilings against arithmetic intensity:
a diagonal bandwidth ceiling that rises with intensity, meeting a flat
compute ceiling at the ridge point. The ridge sits at peak compute
divided by peak bandwidth: about 295 operations per byte for an H100
in FP16. Left of the ridge you are memory-bound; right of it you are
compute-bound.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson builds the roofline for an H100-style device (989 TFLOPS
FP16, 3.35 TB/s), prints the attainable FLOP/s at intensities from
0.1 to 10,000, and classifies two operations: an elementwise vector
add (intensity about 0.08, memory-bound) and a 4096-cubed FP16 matmul
(intensity about 1365, compute-bound). Full mode on CUDA also times a
real 4096 by 4096 matmul and a real copy, and reports achieved TFLOPS
and achieved GB/s against the ceilings.

## Expected observations

- Attainable FLOP/s climbs with intensity until the ridge, then goes
  flat: the flat part is the compute ceiling.
- The elementwise add lands far below the ridge; the matmul lands far
  above it.
- On GPU, the measured matmul's FLOP/s stands orders of magnitude
  above the measured copy's, because a copy does almost no arithmetic
  per byte moved. On laptop GPUs both sit at a small percentage of
  the H100 ceilings; what matters is the contrast between the two
  measured numbers, not their distance from the marketing peak.

## Metric

`ridge_intensity_ops_per_byte`: the break-even arithmetic intensity of
the configured device, with the attainable FLOP/s table and the two op
classifications printed alongside.

## Sources

- Virk, Inference Engineering, chapter "The bottleneck math":
  https://www.arjunvirk.com/inference-engineering.html

## Hardware notes

Smoke mode runs on CPU and does the arithmetic only; no device timing
happens. Full mode without CUDA prints the same table and skips the
timing. Full mode with CUDA times real kernels with
`torch.cuda.synchronize()`.

## Reflection prompts

1. What does the ridge point separate, physically?
2. Why does buying more FLOP/s not help a memory-bound kernel?
3. Which side of the ridge does decode live on, and which does
   prefill?
4. How would you move a memory-bound kernel rightward toward the
   ridge?