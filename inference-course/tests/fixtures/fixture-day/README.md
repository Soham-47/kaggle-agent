# Day 0 — Fixture day

## Goal

Demonstrate the day-folder contract so that automated checks have a known-good example.

## Prerequisites

Python 3.10 or newer. No packages beyond the course environment.

## Concept

A valid day folder holds exactly four files, its README has the required
sections in order, and its lesson prints one concrete metric. This folder
is that contract, made concrete.

## Experiment

Run the lesson:

```
python lesson.py --smoke
```

## Expected observations

The lesson prints one measured metric value on one line.

## Metric

`ttft_ms`: simulated time to first token in milliseconds.

## Sources

None; this is a synthetic example, not a lesson.

## Hardware notes

Runs on CPU. No GPU path exists.

## Reflection prompts

1. What makes a day folder machine-checkable?
2. Why does the lesson accept a `--smoke` flag?
3. What must the Metric section state?
