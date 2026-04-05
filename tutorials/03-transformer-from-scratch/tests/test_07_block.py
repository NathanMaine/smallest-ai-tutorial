"""
Tests for 07_transformer_block.py

Chapter 7 — Transformer Block: one complete Pre-Norm layer from scratch.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_07_block.py -v
"""

import importlib
import sys
import os

# ---------------------------------------------------------------------------
# Path setup — point at the level-c-reader directory
# ---------------------------------------------------------------------------
LEVEL_C = os.path.join(
    os.path.dirname(__file__), '..', '..', 'phase1-from-scratch', 'level-c-reader'
)
sys.path.insert(0, LEVEL_C)

block_mod = importlib.import_module('07_transformer_block')
TransformerBlock = block_mod.TransformerBlock

attn_mod = importlib.import_module('03_self_attention')
create_causal_mask = attn_mod.create_causal_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(seq_len, embed_dim, seed=99):
    import random
    rng = random.Random(seed)
    return [[rng.gauss(0, 0.5) for _ in range(embed_dim)] for _ in range(seq_len)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_block_output_shape():
    """TransformerBlock preserves sequence shape: [3 x 16] → [3 x 16]."""
    embed_dim = 16
    num_heads = 4
    seq_len   = 3

    block = TransformerBlock(embed_dim, num_heads, seed=42)
    x     = _make_input(seq_len, embed_dim)
    out   = block.forward(x)

    assert len(out) == seq_len, (
        f"Expected {seq_len} output positions, got {len(out)}"
    )
    for i, vec in enumerate(out):
        assert len(vec) == embed_dim, (
            f"Position {i}: expected embed_dim={embed_dim}, got {len(vec)}"
        )


def test_block_deterministic():
    """Same input produces identical output on repeated calls."""
    embed_dim = 16
    num_heads = 4
    seq_len   = 4

    block = TransformerBlock(embed_dim, num_heads, seed=0)
    x     = _make_input(seq_len, embed_dim)

    out1 = block.forward(x)
    out2 = block.forward(x)

    for i in range(seq_len):
        for j in range(embed_dim):
            assert out1[i][j] == out2[i][j], (
                f"Position [{i}][{j}]: got {out1[i][j]} then {out2[i][j]}"
            )


def test_block_transforms_input():
    """Block output is not identical to the input (it actually transforms data)."""
    embed_dim = 16
    num_heads = 4
    seq_len   = 3

    block = TransformerBlock(embed_dim, num_heads, seed=7)
    x     = _make_input(seq_len, embed_dim)
    out   = block.forward(x)

    # At least one value must differ from the corresponding input value
    any_different = any(
        abs(out[i][j] - x[i][j]) > 1e-9
        for i in range(seq_len)
        for j in range(embed_dim)
    )
    assert any_different, "Block output is identical to input — no transformation occurred"


def test_block_with_mask():
    """TransformerBlock works correctly with a causal mask."""
    embed_dim = 16
    num_heads = 4
    seq_len   = 5

    block = TransformerBlock(embed_dim, num_heads, seed=42)
    x    = _make_input(seq_len, embed_dim)
    mask = create_causal_mask(seq_len)
    out  = block.forward(x, mask)

    # Shape must still be preserved
    assert len(out) == seq_len, (
        f"Expected {seq_len} positions with mask, got {len(out)}"
    )
    for i, vec in enumerate(out):
        assert len(vec) == embed_dim, (
            f"Position {i} with mask: expected embed_dim={embed_dim}, got {len(vec)}"
        )

    # Output with mask should differ from output without mask
    out_nomask = block.forward(x, mask=None)
    any_different = any(
        abs(out[i][j] - out_nomask[i][j]) > 1e-9
        for i in range(seq_len)
        for j in range(embed_dim)
    )
    assert any_different, (
        "Masked and unmasked outputs are identical — mask appears to have no effect"
    )
