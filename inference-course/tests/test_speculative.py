import numpy as np
import pytest

from shared.speculative import acceptance_rate, expected_emitted


def test_identical_distributions_fully_accepted():
    probs = np.array([0.2, 0.3, 0.5])
    rate = acceptance_rate(probs, probs, n_drafts=4, n_steps=50, rng=np.random.default_rng(7))
    assert rate == pytest.approx(1.0)


def test_disjoint_distributions_all_rejected():
    draft = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    rate = acceptance_rate(draft, target, n_drafts=4, n_steps=50, rng=np.random.default_rng(7))
    assert rate == pytest.approx(0.0)


def test_partial_acceptance_between_zero_and_one():
    draft = np.array([0.5, 0.5, 0.0])
    target = np.array([0.25, 0.25, 0.5])
    rate = acceptance_rate(draft, target, n_drafts=4, n_steps=200, rng=np.random.default_rng(42))
    assert 0.0 < rate < 1.0


def test_acceptance_rate_is_deterministic_with_seed():
    draft = np.array([0.5, 0.5, 0.0])
    target = np.array([0.25, 0.25, 0.5])
    a = acceptance_rate(draft, target, 4, 200, np.random.default_rng(42))
    b = acceptance_rate(draft, target, 4, 200, np.random.default_rng(42))
    assert a == b


def test_expected_emitted_formula():
    assert expected_emitted(3, 0.5) == pytest.approx(1.875)
    assert expected_emitted(3, 1.0) == pytest.approx(4.0)
    assert expected_emitted(3, 0.0) == pytest.approx(1.0)
    assert expected_emitted(0, 0.5) == pytest.approx(1.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        acceptance_rate([1.0], [1.0], n_drafts=0, n_steps=5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        acceptance_rate([1.0], [1.0], n_drafts=2, n_steps=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        expected_emitted(3, 1.5)
    with pytest.raises(ValueError):
        expected_emitted(-1, 0.5)