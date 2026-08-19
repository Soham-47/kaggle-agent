# Day 4 — Requirements: interface, latency budget, unit economics, usage pattern

## Goal

Turn a vague product ask into a concrete serving spec: interface,
latency budget, unit economics, and usage pattern, with the arithmetic
that goes with each.

## Prerequisites

Python 3.10 or newer. The course environment. No downloads and no
network access are needed.

## Concept

Before touching a GPU, pin down four things:

- Interface: streaming chat, a blocking tool call inside an agent, or a
  batch job. Each one changes which metrics matter.
- Latency budget: what "too slow" means to a user. TTFT for streaming,
  total response time for blocking calls.
- Unit economics: the cost per request, per user, or per month that
  keeps the product viable.
- Usage pattern: steady or spiky, and the mix of tiny and giant
  requests.

"Make it fast" is unanswerable. "P95 TTFT under 400 ms at 50 requests
per second for under $2 per million tokens on inputs averaging 2k
tokens" is a spec you can engineer toward. Vague requirements are how
you end up optimizing the wrong axis.

## Experiment

Run the lesson:

```
python lesson.py --smoke
python lesson.py
```

The lesson is a requirements calculator for a fixed spec: 10 requests
per second steady with a 3x business-hour spike, 1500 input plus 300
output tokens per request, a per-token price of $1.20 per million
tokens, a TTFT budget of 400 ms, an ITL budget of 20 ms, an engine that
prefills at 8000 tokens per second with a 12 ms ITL, and revenue of
$0.002 per request. It prints tokens per day, cost per day, required
decode TPS at peak, the TTFT and ITL budget checks, and the margin per
request. Full mode adds a second, more expensive engine and compares.

## Expected observations

- Cost per day is a per-token number times a per-day traffic number:
  unit economics follows usage pattern.
- The required decode TPS at peak is three times the steady-state
  number; capacity must be planned for the peak, not the average.
- The TTFT check comes straight from the prefill rate: 1500 tokens at
  8000 tokens per second is about 188 ms, inside the 400 ms budget.
- The margin per request is negative: at $1.20 per million tokens and
  1800 tokens per request, the request costs more than the $0.002 in
  revenue. The spec is not viable as priced; that is the point of
  unit economics.

## Metric

`cost_per_day_usd`: estimated daily spend on inference tokens for the
spec, with `required_decode_tps_peak` printed alongside.

## Sources

- Virk, Inference Engineering, chapter "Know your requirements":
  https://www.arjunvirk.com/inference-engineering.html

## Hardware notes

Runs on CPU in both modes; the lesson is a deterministic calculator, so
no GPU path exists.

## Reflection prompts

1. Why does a blocking agent tool call not care about TTFT?
2. Which of the four requirements changes if traffic doubles at night
   instead of midday?
3. Where does the margin per request come from, and why does it set a
   ceiling on model cost?
4. What makes a requirement "concrete" enough to engineer toward?