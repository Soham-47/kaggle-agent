# Exercise

Add a second usage pattern: an overnight batch job that sends 2 million
requests between midnight and 6 am, and report its cost per day.

Acceptance hints:

- Add a `BATCH_REQUESTS` constant (2_000_000) and a batch window of
  6 hours.
- Cost per day for the batch is requests times tokens per request
  times price per million.
- Print `batch_cost_per_day_usd=...` and the batch's required average
  TPS over the window.
- Run `python lesson.py --smoke`; the output includes the new lines
  and the day still passes `scripts/check_day.py --day 4 --smoke`.