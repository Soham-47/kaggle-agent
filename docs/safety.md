# Safety model

External mutations follow this sequence:

```text
durable intent → external mutation → durable result or reconciliation
```

Timeout does not prove that a mutation failed. Kernel and submission outbox
entries remain pending until authoritative Kaggle state resolves them. Browser
submission is not an allowed fallback.

Autonomous repairs are constrained by clean-generation requirements, protected
paths and semantics, dependency and diff budgets, static safety scans, test
integrity checks, deterministic verification, independent review, and durable
repair budgets. The repair agent cannot read secrets, push Git, install
dependencies, or alter the acceptance policy.
