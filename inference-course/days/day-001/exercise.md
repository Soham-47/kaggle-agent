# Exercise

Make the lesson queue ten requests with a small artificial delay between
arrivals, then print the p90 of the queued latencies.

Acceptance hints:

- Keep the existing single-request timing; add the queued-request
  timing beside it.
- Use `shared.metrics.percentile` for the p90.
- The printed line names the metric, for example
  `queued_latency_p90_ms=...`.
- Run `python lesson.py --smoke`; the output includes the new line and
  the day still passes `scripts/check_day.py --day 1 --smoke`.