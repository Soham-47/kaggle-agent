# Architecture

Kaggle Agent runs a durable stage pipeline for one configured competition at a
time: research, plan, code, local smoke, kernel train, validation, approval,
submission, feedback, and healing.

The shared runtime keeps stage inputs and outputs in the configured mutable
state root. Competition code lives under `competitions/<id>/` and is selected
by the competition contract rather than by shared defaults.

The optional supervisor owns worker lifecycle, incidents, repair worktrees,
verification, review, generation promotion, rollback, and resume. It is a
separate OS process and does not replace the canonical stage ledger or outbox.
