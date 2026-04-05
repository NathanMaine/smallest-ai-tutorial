"""
Tests for Chapter 3 — Vanishing Gradients.

Verifies:
  - measure_gradient_flow returns the correct number of ratios
  - Ratios are positive for short sequences (gradients exist)
  - Early ratios are smaller than late ratios for long sequences
    (demonstrating the vanishing gradient phenomenon)
  - The ratio list length equals seq_length - 1
  - Edge case: seq_length=2 returns a single ratio
"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '01-mlp-from-scratch', 'solution'))

vg_mod = importlib.import_module('03_vanishing_gradients')
measure_gradient_flow = vg_mod.measure_gradient_flow


# ---------------------------------------------------------------------------
# Shape / type tests
# ---------------------------------------------------------------------------

def test_gradient_flow_returns_ratios():
    """measure_gradient_flow returns seq_length - 1 ratios."""
    ratios = measure_gradient_flow(seq_length=5, hidden_size=4, seed=42)
    assert len(ratios) == 4  # seq_length - 1 ratios


def test_gradient_flow_returns_list():
    """Return type is a list of floats."""
    ratios = measure_gradient_flow(seq_length=5, hidden_size=4, seed=42)
    assert isinstance(ratios, list)
    for r in ratios:
        assert isinstance(r, float)


def test_ratio_length_matches_seq_length_minus_one():
    """Ratio list length is always seq_length - 1."""
    for seq_len in [2, 3, 10, 15]:
        ratios = measure_gradient_flow(seq_length=seq_len, hidden_size=4, seed=7)
        assert len(ratios) == seq_len - 1, (
            f"Expected {seq_len - 1} ratios for seq_length={seq_len}, "
            f"got {len(ratios)}"
        )


# ---------------------------------------------------------------------------
# Short sequence: gradients must be positive
# ---------------------------------------------------------------------------

def test_short_sequence_gradients_exist():
    """All ratios are > 0 for a short sequence."""
    ratios = measure_gradient_flow(seq_length=3, hidden_size=4, seed=42)
    for r in ratios:
        assert r > 0, f"Expected ratio > 0, got {r}"


def test_ratios_are_non_negative():
    """All ratios are >= 0 (gradient norms are non-negative)."""
    ratios = measure_gradient_flow(seq_length=8, hidden_size=6, seed=99)
    for r in ratios:
        assert r >= 0.0, f"Negative ratio: {r}"


# ---------------------------------------------------------------------------
# Long sequence: vanishing gradient behaviour
# ---------------------------------------------------------------------------

def test_long_sequence_gradients_shrink():
    """For long sequences, early gradient ratios are smaller than late ones."""
    ratios = measure_gradient_flow(seq_length=20, hidden_size=8, seed=42)
    # Early steps should have smaller gradient ratios than later steps
    assert ratios[0] < ratios[-1] or max(ratios[:5]) < max(ratios[-5:]), (
        "Expected early ratios to be smaller than late ratios for seq_length=20. "
        f"First ratio: {ratios[0]:.4f}, Last ratio: {ratios[-1]:.4f}"
    )


def test_very_long_sequence_clear_vanishing():
    """For a length-50 sequence, earliest ratios are much smaller than latest."""
    ratios = measure_gradient_flow(seq_length=50, hidden_size=8, seed=42)
    # Average of first quarter vs last quarter
    quarter = max(1, len(ratios) // 4)
    early_avg = sum(ratios[:quarter]) / quarter
    late_avg = sum(ratios[-quarter:]) / quarter
    assert early_avg < late_avg, (
        f"Expected early_avg ({early_avg:.4f}) < late_avg ({late_avg:.4f}) "
        "for seq_length=50 — vanishing gradient not observed."
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_ratios():
    """Same arguments produce identical results (deterministic)."""
    r1 = measure_gradient_flow(seq_length=10, hidden_size=6, seed=123)
    r2 = measure_gradient_flow(seq_length=10, hidden_size=6, seed=123)
    assert r1 == r2, "Results should be identical for the same seed"


def test_different_seeds_different_ratios():
    """Different seeds produce different results."""
    r1 = measure_gradient_flow(seq_length=10, hidden_size=6, seed=1)
    r2 = measure_gradient_flow(seq_length=10, hidden_size=6, seed=2)
    assert r1 != r2, "Different seeds should produce different results"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_minimal_sequence_length_two():
    """seq_length=2 returns exactly 1 ratio."""
    ratios = measure_gradient_flow(seq_length=2, hidden_size=4, seed=42)
    assert len(ratios) == 1


def test_larger_hidden_size():
    """Works correctly with a larger hidden size."""
    ratios = measure_gradient_flow(seq_length=10, hidden_size=16, seed=42)
    assert len(ratios) == 9
    for r in ratios:
        assert r >= 0.0
