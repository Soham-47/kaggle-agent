# Day 8 — Sampling: temperature, top-k, top-p, logit bias

## Goal

Implement the standard sampling pipeline on a fixed logit vector and
measure how each control reshapes the distribution: entropy under
temperature, nucleus size under top-p, and the effect of a logit bias.

## Prerequisites

Python 3.10 or newer. The course environment with numpy and torch.
No downloads and no network access are needed.

## Concept

Each decode step produces a logit vector over the vocabulary. The
decoding strategy decides how those scores become one token.

- Temperature scales the logits before softmax. Higher temperature
  flattens the distribution; lower sharpens it. As temperature tends
  to zero the result becomes greedy: always the argmax.
- Top-k keeps only the k most probable tokens and renormalizes.
  k of 1 is greedy.
- Top-p (nucleus) keeps the smallest set whose cumulative probability
  reaches p. It adapts to the distribution instead of fixing a count.
- Logit bias adds a constant to chosen logits, which steers
  probability mass toward or away from specific tokens.

These are inference-time dials, not model weights. The same model with
different settings can behave like a different model.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson draws a fixed 32-token logit vector from a seeded generator
(64 tokens in full mode), then computes: softmax probabilities at
temperatures 0.5, 1.0, and 1.5 with their entropies; the nucleus size
at top-p 0.9; the kept set at top-k 5; the probability change from a
+3 logit bias on one token; and the diversity of 200 seeded samples
at temperature 1.0 against greedy draws. The first printed line always
starts with `mode=smoke` or `mode=full`.

## Expected observations

- Entropy rises with temperature: the distribution flattens.
- Top-p 0.9 keeps a small nucleus while top-k always keeps exactly 5
  tokens, even when the mass is concentrated on fewer.
- The biased token's probability jumps well above its natural share.
- Greedy draws hit one token every time; temperature-1 draws spread
  across several, so diversity is higher.

## Metric

`entropy_nats_t1`: the Shannon entropy in nats of the softmax
distribution at temperature 1.0, with the other temperatures, the
nucleus size, and the bias delta printed alongside.

## Sources

- aman.ai, Token Sampling Methods primer:
  https://aman.ai/primers/ai/token-sampling/

## Hardware notes

Runs on CPU in both modes; the experiment is numpy arithmetic, so no
GPU path exists. Full mode only widens the logit vector and the sample
count.

## Reflection prompts

1. Why does temperature act on logits before softmax rather than on
   probabilities?
2. When is top-p better than top-k, and when is it worse?
3. What does a logit bias buy you in a structured-output setting?
4. Why is greedy decoding deterministic and temperature-1 sampling
   not, even with the same seed and logits?