# Exercise

Add a batch size of 128 to the sweep and report the latency/throughput
ratio for each batch.

Acceptance hints:

- Append 128 to the batch list in both modes.
- Print the ratio `latency_ms / throughput_req_per_s` for every batch.
- Expect the ratio to fall as the batch grows.
- Run `python lesson.py --smoke`; the output includes batch 128 and the
  day still passes `scripts/check_day.py --day 2 --smoke`.