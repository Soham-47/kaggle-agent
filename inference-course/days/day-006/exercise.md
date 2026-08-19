# Exercise

Add a second user turn so the template holds a two-turn conversation,
and report the overhead of the second turn.

Acceptance hints:

- Extend `apply_chat_template` with a `history` argument holding the
  first assistant reply.
- The template ends with a second `user` turn followed by the
  assistant marker.
- Print `second_turn_overhead_tokens=...`.
- Run `python lesson.py --smoke`; the output includes the new line
  and the day still passes `scripts/check_day.py --day 6 --smoke`.