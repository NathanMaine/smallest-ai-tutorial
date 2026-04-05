"""
Tests for 06_layer_norm_residual.py

Chapter 6 — Layer Norm + Residual: training stability from scratch.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_06_layernorm.py -v
"""

import importlib
import sys
import os
import math

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)
ln_mod = importlib.import_module('06_layer_norm_residual')
LayerNorm    = ln_mod.LayerNorm
residual_add = ln_mod.residual_add


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_layernorm_output_shape():
    """LayerNorm output has the same shape as input."""
    dim = 8
    ln = LayerNorm(dim)
    x = [float(i) for i in range(dim)]
    out = ln.forward(x)
    assert len(out) == dim, (
        f"Expected output length {dim}, got {len(out)}"
    )


def test_layernorm_mean_near_zero():
    """After normalization, mean of output is approximately 0 (with default gamma=1, beta=0)."""
    dim = 16
    ln = LayerNorm(dim)
    x = [float(i * 3 + 1) for i in range(dim)]  # non-trivial input
    out = ln.forward(x)
    mean = sum(out) / len(out)
    assert abs(mean) < 1e-5, (
        f"Expected mean ≈ 0, got {mean}"
    )


def test_layernorm_var_near_one():
    """After normalization, variance of output is approximately 1 (with default gamma=1, beta=0)."""
    dim = 16
    ln = LayerNorm(dim)
    x = [float(i * 3 + 1) for i in range(dim)]
    out = ln.forward(x)
    mean = sum(out) / len(out)
    var = sum((v - mean) ** 2 for v in out) / len(out)
    assert abs(var - 1.0) < 1e-4, (
        f"Expected variance ≈ 1.0, got {var}"
    )


def test_layernorm_sequence():
    """forward_sequence() applies LayerNorm to each vector in the sequence."""
    dim = 8
    seq_len = 4
    ln = LayerNorm(dim)
    sequence = [[float(i + j) for i in range(dim)] for j in range(seq_len)]
    out = ln.forward_sequence(sequence)

    assert len(out) == seq_len, (
        f"Expected {seq_len} output vectors, got {len(out)}"
    )
    for i, vec in enumerate(out):
        assert len(vec) == dim, (
            f"Output vector {i} has length {len(vec)}, expected {dim}"
        )
        # Each normalized vector should have mean ≈ 0
        mean = sum(vec) / len(vec)
        assert abs(mean) < 1e-5, (
            f"Sequence position {i}: expected mean ≈ 0, got {mean}"
        )


def test_residual_add():
    """residual_add correctly adds two vector lists element-wise."""
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    sublayer = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    out = residual_add(x, sublayer)

    expected = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
    for i, (got_vec, exp_vec) in enumerate(zip(out, expected)):
        for j, (got, exp) in enumerate(zip(got_vec, exp_vec)):
            assert abs(got - exp) < 1e-9, (
                f"Position [{i}][{j}]: expected {exp}, got {got}"
            )


def test_residual_preserves_shape():
    """residual_add output shape matches input shape."""
    seq_len = 5
    dim = 12
    x = [[float(i) for i in range(dim)] for _ in range(seq_len)]
    sublayer = [[float(i * 0.1) for i in range(dim)] for _ in range(seq_len)]
    out = residual_add(x, sublayer)

    assert len(out) == seq_len, (
        f"Expected {seq_len} output vectors, got {len(out)}"
    )
    for i, vec in enumerate(out):
        assert len(vec) == dim, (
            f"Output vector {i} has length {len(vec)}, expected {dim}"
        )
