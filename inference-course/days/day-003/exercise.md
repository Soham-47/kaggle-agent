# Exercise

Add a p99-to-p50 ratio to the report and make the prefill distribution
worse for 1 in 20 requests (a long prefill tail).

Acceptance hints:

- Give 5% of the simulated requests a prefill five times longer than
  the rest.
- Print `ttft_p99_to_p50_ratio=...`.
- Watch the ratio jump compared to the run without the tail.
- Run `python lesson.py --smoke`; the output includes the new line and
  the day still passes `scripts/check_day.py --day 3 --smoke`.