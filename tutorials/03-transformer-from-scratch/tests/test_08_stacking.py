"""
Tests for 08_stacking_layers.py

Chapter 8 — Stacking Layers: the full multi-layer Transformer from scratch.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_08_stacking.py -v
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

stack_mod   = importlib.import_module('08_stacking_layers')
Transformer = stack_mod.Transformer

embed_mod  = importlib.import_module('01_embeddings')
Vocabulary = embed_mod.Vocabulary


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 20
EMBED_DIM  = 16
NUM_HEADS  = 4
NUM_LAYERS = 2


def _small_transformer(seed=42):
    return Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=32,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_transformer_creation():
    """Transformer can be instantiated without errors."""
    model = _small_transformer()
    assert model is not None
    assert len(model.blocks) == NUM_LAYERS
    assert model.vocab_size == VOCAB_SIZE
    assert model.embed_dim == EMBED_DIM


def test_transformer_forward_shape():
    """Forward pass on 5 tokens returns [5 × vocab_size] logits."""
    model       = _small_transformer()
    token_ids   = [0, 1, 2, 3, 4]   # arbitrary token indices within range
    logits      = model.forward(token_ids)

    assert len(logits) == len(token_ids), (
        f"Expected {len(token_ids)} positions, got {len(logits)}"
    )
    for i, pos_logits in enumerate(logits):
        assert len(pos_logits) == VOCAB_SIZE, (
            f"Position {i}: expected vocab_size={VOCAB_SIZE} logits, got {len(pos_logits)}"
        )


def test_transformer_different_inputs():
    """Different token sequences produce different logit outputs."""
    model    = _small_transformer()
    ids_a    = [0, 1, 2, 3]
    ids_b    = [4, 5, 6, 7]

    logits_a = model.forward(ids_a)
    logits_b = model.forward(ids_b)

    # At least one logit value should differ between the two sequences
    any_different = any(
        abs(logits_a[i][v] - logits_b[i][v]) > 1e-9
        for i in range(min(len(logits_a), len(logits_b)))
        for v in range(VOCAB_SIZE)
    )
    assert any_different, (
        "Different token inputs produced identical outputs — model appears insensitive to input"
    )


def test_transformer_param_count():
    """get_params_count() returns a positive integer."""
    model       = _small_transformer()
    param_count = model.get_params_count()

    assert isinstance(param_count, int), (
        f"Expected int, got {type(param_count)}"
    )
    assert param_count > 0, (
        f"Expected positive parameter count, got {param_count}"
    )


def test_transformer_handles_short_sequence():
    """Transformer works correctly with a single-token sequence."""
    model    = _small_transformer()
    token_ids = [3]   # single token
    logits   = model.forward(token_ids)

    assert len(logits) == 1, (
        f"Expected 1 output position, got {len(logits)}"
    )
    assert len(logits[0]) == VOCAB_SIZE, (
        f"Expected {VOCAB_SIZE} logits, got {len(logits[0])}"
    )
