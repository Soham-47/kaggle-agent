# Exercise

Recompute the roofline for a hypothetical device with half the
bandwidth, and report how the ridge and the elementwise classification
change.

Acceptance hints:

- Use `PEAK_BW / 2` and the same compute ceiling.
- Print `ridge_halved_bw=...`.
- Expect the ridge to double and the elementwise add to stay
  memory-bound.
- Run `python lesson.py --smoke`; the output includes the new line
  and the day still passes `scripts/check_day.py --day 9 --smoke`.