# Day 5 — Model selection and evals; shared vs dedicated

## Goal

Run a seeded evaluation over two candidate models, compare their
measured quality and latency, and price the same traffic on a shared
per-token API versus a dedicated GPU.

## Prerequisites

Python 3.10 or newer. The course environment with numpy and torch.
No downloads and no network access are needed.

## Concept

Evals first, always. Pick the smallest model that clears your quality
bar on your task, not the one topping a generic leaderboard. Quality is
a property of a model on your workload, measured with your eval set.

Shared versus dedicated is a pricing and capacity choice, not a model
license choice. Shared inference bills per token on capacity shared
with strangers. Dedicated inference bills per GPU-hour on capacity that
is yours. You can run an open model on a shared endpoint or pay for a
dedicated deployment. At scale, a dedicated optimized deployment is
often far cheaper than per-token API pricing, because you stop paying
someone else's margin on idle capacity.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson defines two candidates: model A (small) with a true
per-item accuracy of 0.86 and a mean latency of 80 ms, and model B
(large) with 0.93 and 140 ms. A seeded generator runs a Monte Carlo
eval over 80 items (200 in full mode): each item is "correct" with the
model's probability, and each request draws a lognormal latency. The
lesson reports measured accuracy and p95 latency for both, then prices
5 requests per second of 1200-token traffic on a shared API at $1.50
per million tokens and on dedicated GPUs at $2.20 (A) and $3.90 (B)
per hour.

## Expected observations

- Measured accuracy lands near the true value: the eval recovers the
  model's quality.
- Model B's accuracy gain is small in absolute terms (0.07) while its
  latency and hourly cost are much larger.
- The shared API cost per day is per-token and identical for both
  models; the dedicated cost differs and is much lower at this
  traffic level.

## Metric

`eval_accuracy_b`: the measured accuracy of the larger model on the
synthetic eval, with `eval_accuracy_a`, both p95 latencies, and both
daily costs printed alongside.

## Sources

- Virk, Inference Engineering, chapter "Model selection" and "Shared
  vs dedicated":
  https://www.arjunvirk.com/inference-engineering.html

## Hardware notes

Runs on CPU in both modes; the eval and the pricing are synthetic, so
no GPU path exists. Full mode only increases the eval size and adds
nothing else.

## Reflection prompts

1. Why is a generic leaderboard score a weak reason to pick a model?
2. Shared and dedicated are orthogonal to open and closed. Give one
   example of each combination.
3. At what traffic level does the dedicated deployment stop being
   cheaper? How would you find that point?
4. What would you change in the eval to make it fit a text-to-SQL
   workload?