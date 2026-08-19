"""Simulations for speculative decoding experiments.

acceptance_rate() runs rejection-style draft-target sampling over many
iterations; expected_emitted() gives the classic expected-tokens-per-
iteration formula for a fixed acceptance rate.
"""

import numpy as np


def acceptance_rate(draft_probs, target_probs, n_drafts: int, n_steps: int, rng) -> float:
    """Return the simulated draft-token acceptance rate over n_steps iterations.

    Each iteration draws n_drafts tokens from draft_probs and accepts a
    proposed token with probability min(1, target / draft) for that token.
    rng must be a numpy.random.Generator, e.g. np.random.default_rng(seed).
    """
    if n_drafts < 1:
        raise ValueError("n_drafts must be at least 1")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    draft = np.asarray(draft_probs, dtype=float)
    target = np.asarray(target_probs, dtype=float)
    if draft.shape != target.shape:
        raise ValueError("draft_probs and target_probs must have the same shape")
    if not np.allclose(draft.sum(), 1.0, atol=1e-6) or not np.allclose(target.sum(), 1.0, atol=1e-6):
        raise ValueError("probability vectors must sum to 1")
    if np.any(draft < 0) or np.any(target < 0):
        raise ValueError("probabilities must not be negative")
    accepted = 0
    for _ in range(n_steps):
        proposals = rng.choice(draft.size, size=n_drafts, p=draft)
        for token in proposals:
            if rng.random() < min(1.0, target[token] / draft[token]):
                accepted += 1
    return accepted / (n_steps * n_drafts)


def expected_emitted(n_drafts: int, acceptance: float) -> float:
    """Return the expected tokens emitted per iteration for a draft block.

    acceptance is the per-token acceptance rate. The result counts the
    bonus token the target produces when the whole draft block is
    accepted.
    """
    if n_drafts < 0:
        raise ValueError("n_drafts must not be negative")
    if not 0.0 <= acceptance <= 1.0:
        raise ValueError("acceptance must be between 0 and 1")
    if acceptance == 1.0:
        return float(n_drafts + 1)
    return (1.0 - acceptance ** (n_drafts + 1)) / (1.0 - acceptance)