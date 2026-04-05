"""
Tests for 05_feed_forward.py

Chapter 5 — Feed-Forward Network: position-wise FFN from scratch.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_05_ffn.py -v
"""

import importlib
import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)
ffn_mod = importlib.import_module('05_feed_forward')
FeedForward = ffn_mod.FeedForward


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ffn_output_shape():
    """Input [3 x 8] produces output [3 x 8]."""
    embed_dim = 8
    seq_len = 3
    ff = FeedForward(embed_dim, seed=42)

    # Create simple input: seq_len vectors of length embed_dim
    x = [[float(i + j * 0.1) for i in range(embed_dim)] for j in range(seq_len)]
    out = ff.forward(x)

    assert len(out) == seq_len, (
        f"Expected {seq_len} output vectors, got {len(out)}"
    )
    for i, vec in enumerate(out):
        assert len(vec) == embed_dim, (
            f"Output vector {i} has length {len(vec)}, expected {embed_dim}"
        )


def test_ffn_different_positions_independent():
    """Changing one position's input does not affect other positions' outputs."""
    embed_dim = 8
    seq_len = 3
    ff = FeedForward(embed_dim, seed=42)

    x = [[float(i + j * 0.1) for i in range(embed_dim)] for j in range(seq_len)]

    out_original = ff.forward(x)

    # Change position 0 only
    x_modified = [list(v) for v in x]
    x_modified[0] = [99.0] * embed_dim

    out_modified = ff.forward(x_modified)

    # Position 0 should change
    assert out_original[0] != out_modified[0], (
        "Changing position 0 input should change position 0 output"
    )

    # Positions 1 and 2 should be identical
    assert out_original[1] == out_modified[1], (
        "Changing position 0 input should NOT affect position 1 output"
    )
    assert out_original[2] == out_modified[2], (
        "Changing position 0 input should NOT affect position 2 output"
    )


def test_ffn_deterministic():
    """Same input always produces the same output."""
    embed_dim = 8
    seq_len = 3
    ff = FeedForward(embed_dim, seed=42)

    x = [[float(i + j * 0.1) for i in range(embed_dim)] for j in range(seq_len)]

    out1 = ff.forward(x)
    out2 = ff.forward(x)

    assert out1 == out2, "Same input should always produce the same output"


def test_ffn_default_ff_dim():
    """ff_dim defaults to 4 * embed_dim."""
    embed_dim = 8
    ff = FeedForward(embed_dim, seed=42)

    assert ff.ff_dim == 4 * embed_dim, (
        f"Expected ff_dim={4 * embed_dim}, got {ff.ff_dim}"
    )
    # Also verify W1 has correct shape: [ff_dim x embed_dim]
    assert len(ff.W1) == ff.ff_dim, (
        f"W1 should have {ff.ff_dim} rows (ff_dim), got {len(ff.W1)}"
    )
    assert len(ff.W1[0]) == embed_dim, (
        f"W1 rows should have length {embed_dim} (embed_dim), got {len(ff.W1[0])}"
    )
