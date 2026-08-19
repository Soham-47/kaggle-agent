# Exercise

Generate from two different context lengths and report how the decode
time per token changes when the context grows.

Acceptance hints:

- Prefill at ctx lengths 64 and 128 (smoke mode) and decode the same
  number of tokens from both.
- Print `decode_ms_per_token_ctx64=...` and
  `decode_ms_per_token_ctx128=...`.
- In a real model the per-token time grows with context because each
  step reads the growing KV cache; in this tiny GRU it stays flat,
  which is worth noting in one sentence.
- Run `python lesson.py --smoke`; the output includes the new lines
  and the day still passes `scripts/check_day.py --day 7 --smoke`.