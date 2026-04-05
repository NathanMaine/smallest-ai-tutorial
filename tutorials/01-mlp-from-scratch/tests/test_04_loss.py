"""Tests for Chapter 4: Loss Functions (04_loss_function.py)"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss
mse_loss = loss_mod.mse_loss


# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one():
    """softmax([1, 2, 3]) must sum to 1.0."""
    probs = softmax([1.0, 2.0, 3.0])
    assert abs(sum(probs) - 1.0) < 1e-10, f"Expected sum ~1.0, got {sum(probs)}"


def test_softmax_all_positive():
    """Every softmax output must be strictly greater than zero."""
    probs = softmax([1.0, 2.0, 3.0])
    for p in probs:
        assert p > 0, f"Expected all values > 0, got {p}"


def test_softmax_largest_input_largest_output():
    """softmax preserves the ordering: larger input → larger output."""
    logits = [1.0, 2.0, 3.0]
    probs = softmax(logits)
    # logits are strictly increasing, so probs should be too
    assert probs[0] < probs[1] < probs[2], (
        f"Ordering not preserved: {probs}"
    )


def test_softmax_equal_inputs():
    """softmax([1, 1, 1]) should return [1/3, 1/3, 1/3]."""
    probs = softmax([1.0, 1.0, 1.0])
    for p in probs:
        assert abs(p - 1.0 / 3.0) < 1e-10, f"Expected 1/3, got {p}"


def test_softmax_numerical_stability():
    """softmax([1000, 1001, 1002]) must not overflow and must sum to 1."""
    probs = softmax([1000.0, 1001.0, 1002.0])
    assert abs(sum(probs) - 1.0) < 1e-10, f"Expected sum ~1.0, got {sum(probs)}"
    for p in probs:
        assert 0 < p < 1, f"Probability out of range: {p}"


# ---------------------------------------------------------------------------
# Cross-entropy loss tests
# ---------------------------------------------------------------------------

def test_cross_entropy_perfect_prediction():
    """Low loss when the model strongly predicts the correct class."""
    target = [0.0, 0.0, 1.0]
    predicted = [0.01, 0.01, 0.98]
    loss = cross_entropy_loss(predicted, target)
    assert loss < 0.1, f"Expected loss < 0.1 for near-perfect prediction, got {loss}"


def test_cross_entropy_wrong_prediction():
    """High loss when the model confidently predicts the wrong class."""
    target = [0.0, 0.0, 1.0]
    predicted = [0.98, 0.01, 0.01]
    loss = cross_entropy_loss(predicted, target)
    assert loss > 3.0, f"Expected loss > 3.0 for confidently wrong prediction, got {loss}"


def test_cross_entropy_always_positive():
    """Cross-entropy loss must be non-negative for any imperfect prediction."""
    target = [0.0, 1.0, 0.0]
    predicted = [0.1, 0.7, 0.2]
    loss = cross_entropy_loss(predicted, target)
    assert loss > 0, f"Expected loss > 0 for imperfect prediction, got {loss}"


# ---------------------------------------------------------------------------
# MSE loss tests
# ---------------------------------------------------------------------------

def test_mse_perfect():
    """MSE([1, 0, 0], [1, 0, 0]) must be exactly 0."""
    loss = mse_loss([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert loss == 0.0, f"Expected 0.0 for perfect prediction, got {loss}"


def test_mse_known_value():
    """MSE([1, 0], [0, 1]) should equal 1.0.

    (1-0)^2 + (0-1)^2 = 1 + 1 = 2, divided by n=2 → 1.0
    """
    loss = mse_loss([1.0, 0.0], [0.0, 1.0])
    assert abs(loss - 1.0) < 1e-10, f"Expected MSE = 1.0, got {loss}"
