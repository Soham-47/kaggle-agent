# Exercise

Add a top-k-with-top-p pipeline (apply top-k 5 first, then top-p 0.9 on
the survivors) and report the combined nucleus size.

Acceptance hints:

- Apply the two filters in sequence and renormalize after each.
- Print `combined_nucleus_size=...` and compare it to the plain
  top-p nucleus size.
- Expect the combined size to be 5 or fewer tokens.
- Run `python lesson.py --smoke`; the output includes the new line
  and the day still passes `scripts/check_day.py --day 8 --smoke`.