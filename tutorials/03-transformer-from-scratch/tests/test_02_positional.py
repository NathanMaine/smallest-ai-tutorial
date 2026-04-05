"""
Tests for 02_positional_encoding.py

Chapter 2 — Positional Encoding: sinusoidal PE and add_position_info.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_02_positional.py -v
"""

import importlib
import sys
import os
import math

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)
pe_mod = importlib.import_module('02_positional_encoding')

sinusoidal_encoding = pe_mod.sinusoidal_encoding
add_position_info   = pe_mod.add_position_info


# ---------------------------------------------------------------------------
# sinusoidal_encoding tests
# ---------------------------------------------------------------------------

def test_pe_shape():
    """sinusoidal_encoding(10, 16) returns a matrix of shape [10][16]."""
    max_seq_len, embed_dim = 10, 16
    pe = sinusoidal_encoding(max_seq_len, embed_dim)

    assert len(pe) == max_seq_len, (
        f"Expected {max_seq_len} rows, got {len(pe)}"
    )
    for pos, row in enumerate(pe):
        assert len(row) == embed_dim, (
            f"Row {pos}: expected {embed_dim} cols, got {len(row)}"
        )


def test_pe_values_in_range():
    """All positional encoding values lie in [-1.0, 1.0] (sin/cos outputs)."""
    pe = sinusoidal_encoding(20, 32)
    for pos, row in enumerate(pe):
        for d, val in enumerate(row):
            assert -1.0 <= val <= 1.0, (
                f"PE[{pos}][{d}] = {val:.6f} is outside [-1, 1]"
            )


def test_pe_different_positions():
    """Position 0 and position 1 encode to different vectors."""
    pe = sinusoidal_encoding(10, 16)
    assert pe[0] != pe[1], (
        "PE[0] and PE[1] must be different vectors"
    )


def test_pe_deterministic():
    """Two calls with the same arguments return identical matrices."""
    pe1 = sinusoidal_encoding(10, 16)
    pe2 = sinusoidal_encoding(10, 16)
    assert pe1 == pe2, "sinusoidal_encoding must be deterministic"


def test_pe_sin_cos_formula():
    """Spot-check that PE values match the hand-computed formula."""
    max_seq_len, embed_dim = 5, 8
    pe = sinusoidal_encoding(max_seq_len, embed_dim)

    for pos in range(max_seq_len):
        for i in range(embed_dim // 2):
            denom = math.pow(10000.0, (2 * i) / embed_dim)
            angle = pos / denom
            expected_sin = math.sin(angle)
            expected_cos = math.cos(angle)
            assert abs(pe[pos][2 * i]     - expected_sin) < 1e-12, (
                f"PE[{pos}][{2*i}] sin mismatch: "
                f"got {pe[pos][2*i]:.12f}, expected {expected_sin:.12f}"
            )
            assert abs(pe[pos][2 * i + 1] - expected_cos) < 1e-12, (
                f"PE[{pos}][{2*i+1}] cos mismatch: "
                f"got {pe[pos][2*i+1]:.12f}, expected {expected_cos:.12f}"
            )


def test_pe_all_positions_unique():
    """Every row in the PE matrix is unique (no two positions identical)."""
    pe = sinusoidal_encoding(20, 16)
    # Convert each row to a tuple for hashing
    row_tuples = [tuple(row) for row in pe]
    assert len(set(row_tuples)) == len(pe), (
        "Some positions share identical encodings — all positions should be unique"
    )


# ---------------------------------------------------------------------------
# add_position_info tests
# ---------------------------------------------------------------------------

def test_add_position_info_shape():
    """Result has the same shape as the input embeddings."""
    pe         = sinusoidal_encoding(10, 8)
    embeddings = [[0.1 * d for d in range(8)] for _ in range(5)]
    result     = add_position_info(embeddings, pe)

    assert len(result) == len(embeddings), (
        f"Expected {len(embeddings)} rows, got {len(result)}"
    )
    for t, vec in enumerate(result):
        assert len(vec) == len(embeddings[t]), (
            f"Row {t}: expected {len(embeddings[t])} dims, got {len(vec)}"
        )


def test_add_position_info_values_differ():
    """Result values differ from the original embeddings (PE was actually added)."""
    pe         = sinusoidal_encoding(10, 8)
    embeddings = [[float(d) for d in range(8)] for _ in range(5)]
    result     = add_position_info(embeddings, pe)

    # At least one position with a non-zero PE contribution must differ
    any_diff = any(
        result[t][d] != embeddings[t][d]
        for t in range(len(embeddings))
        for d in range(len(embeddings[t]))
    )
    assert any_diff, "add_position_info returned identical values — PE was not added"


def test_add_position_info_zero_embeddings():
    """Adding PE to zero embeddings returns the PE itself."""
    max_seq_len, embed_dim = 5, 8
    pe         = sinusoidal_encoding(max_seq_len, embed_dim)
    zero_embed = [[0.0] * embed_dim for _ in range(max_seq_len)]
    result     = add_position_info(zero_embed, pe)

    for t in range(max_seq_len):
        for d in range(embed_dim):
            assert abs(result[t][d] - pe[t][d]) < 1e-12, (
                f"result[{t}][{d}]={result[t][d]:.12f} != pe[{t}][{d}]={pe[t][d]:.12f}"
            )


def test_add_position_info_correct_elementwise():
    """Verify that result[t][d] == embeddings[t][d] + pe[t][d] for all t, d."""
    pe         = sinusoidal_encoding(10, 8)
    embeddings = [[float(t * 8 + d) for d in range(8)] for t in range(6)]
    result     = add_position_info(embeddings, pe)

    for t in range(len(embeddings)):
        for d in range(len(embeddings[t])):
            expected = embeddings[t][d] + pe[t][d]
            assert abs(result[t][d] - expected) < 1e-12, (
                f"result[{t}][{d}]={result[t][d]:.12f}, "
                f"expected {expected:.12f}"
            )


def test_add_position_info_does_not_mutate_input():
    """add_position_info must not modify the original embeddings list."""
    pe         = sinusoidal_encoding(10, 8)
    embeddings = [[1.0] * 8 for _ in range(5)]
    original   = [list(row) for row in embeddings]   # deep copy
    add_position_info(embeddings, pe)

    assert embeddings == original, (
        "add_position_info mutated the input embeddings list"
    )
