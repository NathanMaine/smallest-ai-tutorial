"""
Tests for 04_multi_head_attention.py

Chapter 4 — Multi-Head Attention: parallel attention heads combined via output projection.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_04_multihead.py -v
"""

import importlib
import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)
mha_mod = importlib.import_module('04_multi_head_attention')

MultiHeadAttention = mha_mod.MultiHeadAttention


# ---------------------------------------------------------------------------
# MultiHeadAttention tests
# ---------------------------------------------------------------------------

def test_multihead_output_shape():
    """embed_dim=16, 4 heads, seq_len=3 -> output [3 x 16]."""
    embed_dim = 16
    num_heads = 4
    mha = MultiHeadAttention(embed_dim, num_heads, seed=42)

    x = [
        [0.1 * (i + j) for j in range(embed_dim)]
        for i in range(3)
    ]

    output = mha.forward(x)

    assert len(output) == 3, f"Expected 3 output vectors, got {len(output)}"
    for i, vec in enumerate(output):
        assert len(vec) == embed_dim, (
            f"Output {i} has dim {len(vec)}, expected {embed_dim}"
        )


def test_multihead_head_count():
    """Model has correct number of heads."""
    embed_dim = 16
    num_heads = 4
    mha = MultiHeadAttention(embed_dim, num_heads, seed=42)

    assert len(mha.heads) == num_heads, (
        f"Expected {num_heads} heads, got {len(mha.heads)}"
    )


def test_multihead_deterministic():
    """Same input produces same output."""
    embed_dim = 8
    num_heads = 2
    mha = MultiHeadAttention(embed_dim, num_heads, seed=42)

    x = [[0.1 * (i + j) for j in range(embed_dim)] for i in range(3)]

    out1 = mha.forward(x)
    out2 = mha.forward(x)

    for v1, v2 in zip(out1, out2):
        for a, b in zip(v1, v2):
            assert abs(a - b) < 1e-10, (
                f"Same input should produce identical output, got {a} vs {b}"
            )


def test_multihead_different_inputs():
    """Different inputs produce different outputs."""
    embed_dim = 8
    num_heads = 2
    mha = MultiHeadAttention(embed_dim, num_heads, seed=42)

    x1 = [[1.0 if j == i else 0.0 for j in range(embed_dim)] for i in range(3)]
    x2 = [[0.5 for _ in range(embed_dim)] for _ in range(3)]

    out1 = mha.forward(x1)
    out2 = mha.forward(x2)

    any_diff = False
    for v1, v2 in zip(out1, out2):
        for a, b in zip(v1, v2):
            if abs(a - b) > 1e-6:
                any_diff = True
                break
        if any_diff:
            break

    assert any_diff, "Different inputs should produce different outputs"
