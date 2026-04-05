"""
Tests for 03_self_attention.py

Chapter 3 — Self-Attention: scaled dot-product attention and SelfAttention class.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_03_attention.py -v
"""

import importlib
import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)
attn_mod = importlib.import_module('03_self_attention')

create_causal_mask            = attn_mod.create_causal_mask
scaled_dot_product_attention  = attn_mod.scaled_dot_product_attention
SelfAttention                 = attn_mod.SelfAttention


# ---------------------------------------------------------------------------
# scaled_dot_product_attention tests
# ---------------------------------------------------------------------------

def test_attention_output_shape():
    """3 query positions, d_k=4 -> 3 output vectors of dim 4."""
    Q = [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]]
    K = [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]]
    V = [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]]

    outputs, weights = scaled_dot_product_attention(Q, K, V)

    assert len(outputs) == 3, f"Expected 3 output vectors, got {len(outputs)}"
    for i, vec in enumerate(outputs):
        assert len(vec) == 4, f"Output {i} has dim {len(vec)}, expected 4"


def test_attention_weights_sum_to_one():
    """Each row of attention weights sums to ~1.0."""
    Q = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0]]

    _, weights = scaled_dot_product_attention(Q, K, V)

    for i, row in enumerate(weights):
        row_sum = sum(row)
        assert abs(row_sum - 1.0) < 1e-6, (
            f"Weight row {i} sums to {row_sum}, expected ~1.0"
        )


# ---------------------------------------------------------------------------
# create_causal_mask tests
# ---------------------------------------------------------------------------

def test_causal_mask_shape():
    """Mask is seq_len x seq_len."""
    mask = create_causal_mask(5)
    assert len(mask) == 5, f"Expected 5 rows, got {len(mask)}"
    for i, row in enumerate(mask):
        assert len(row) == 5, f"Row {i} has length {len(row)}, expected 5"


def test_causal_mask_blocks_future():
    """Position 0 can only see position 0, position 1 can see 0-1, etc."""
    mask = create_causal_mask(4)

    # mask[i][j] = True means j is BLOCKED (j > i means future)
    # Position 0: can see [0], blocked from [1,2,3]
    assert mask[0][0] == False
    assert mask[0][1] == True
    assert mask[0][2] == True
    assert mask[0][3] == True

    # Position 1: can see [0,1], blocked from [2,3]
    assert mask[1][0] == False
    assert mask[1][1] == False
    assert mask[1][2] == True
    assert mask[1][3] == True

    # Position 2: can see [0,1,2], blocked from [3]
    assert mask[2][0] == False
    assert mask[2][1] == False
    assert mask[2][2] == False
    assert mask[2][3] == True

    # Position 3: can see all
    assert mask[3][0] == False
    assert mask[3][1] == False
    assert mask[3][2] == False
    assert mask[3][3] == False


def test_attention_with_mask():
    """Masked attention: position 0 only attends to itself."""
    Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    mask = create_causal_mask(3)
    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    # Position 0 can only attend to position 0
    assert abs(weights[0][0] - 1.0) < 1e-6, (
        f"Position 0 should attend 100% to itself, got weight {weights[0][0]}"
    )
    # Positions 1 and 2 in the future should have ~0 weight for position 0
    assert weights[0][1] < 1e-4, f"Position 0 should not attend to position 1"
    assert weights[0][2] < 1e-4, f"Position 0 should not attend to position 2"


# ---------------------------------------------------------------------------
# SelfAttention class tests
# ---------------------------------------------------------------------------

def test_self_attention_forward():
    """SelfAttention with embed_dim=8, head_dim=4 produces correct shapes."""
    embed_dim = 8
    head_dim  = 4
    sa = SelfAttention(embed_dim, head_dim, seed=42)

    # 3 input vectors of dim 8
    x = [
        [0.1 * (i + j) for j in range(embed_dim)]
        for i in range(3)
    ]

    outputs, attn_weights = sa.forward(x)

    assert len(outputs) == 3, f"Expected 3 output vectors, got {len(outputs)}"
    for i, vec in enumerate(outputs):
        assert len(vec) == head_dim, (
            f"Output {i} has dim {len(vec)}, expected {head_dim}"
        )

    assert len(attn_weights) == 3, f"Expected 3 weight rows, got {len(attn_weights)}"
    for i, row in enumerate(attn_weights):
        assert len(row) == 3, f"Weight row {i} has length {len(row)}, expected 3"


def test_different_inputs_different_attention():
    """Different inputs produce different attention patterns."""
    embed_dim = 8
    head_dim  = 4
    sa = SelfAttention(embed_dim, head_dim, seed=42)

    x1 = [[1.0 if j == i else 0.0 for j in range(embed_dim)] for i in range(3)]
    x2 = [[0.5 for _ in range(embed_dim)] for _ in range(3)]

    out1, w1 = sa.forward(x1)
    out2, w2 = sa.forward(x2)

    # At least one output vector should differ
    any_diff = False
    for v1, v2 in zip(out1, out2):
        for a, b in zip(v1, v2):
            if abs(a - b) > 1e-6:
                any_diff = True
                break
        if any_diff:
            break

    assert any_diff, "Different inputs should produce different attention outputs"
