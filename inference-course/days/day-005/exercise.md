# Exercise

Add a third candidate, a fine-tuned small model with accuracy 0.92 and
mean latency 85 ms, and print the accuracy-per-dollar ranking of all
three.

Acceptance hints:

- Add the candidate to the `CANDIDATES` dict with a reasonable hourly
  GPU price.
- Print `model=c accuracy=...` in the same format as the others.
- Rank by `eval_accuracy / dedicated_cost_per_day`.
- Run `python lesson.py --smoke`; the output includes the new model
  and the day still passes `scripts/check_day.py --day 5 --smoke`.