# Exercise

Quantize the model to INT8 (1 byte per weight) in the intensity
calculation and report how decode intensity changes.

Acceptance hints:

- Set `DTYPE_BYTES = 1` in a second pass over the same sequence
  lengths.
- Print `decode_intensity_int8=...` (expect 2.0) and the new
  classification at seq len 1.
- Note in one sentence why halving the bytes doubles the intensity.
- Run `python lesson.py --smoke`; the output includes the new lines
  and the day still passes `scripts/check_day.py --day 10 --smoke`.