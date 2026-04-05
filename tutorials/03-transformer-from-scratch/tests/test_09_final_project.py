"""
Tests for 09_training.py

Chapter 9 — Training the Transformer: teacher forcing and analytical
output-layer gradients. All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_09_training.py -v
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

train_mod = importlib.import_module('09_final_project')
TransformerTrainer = train_mod.TransformerTrainer
prepare_lm_data = train_mod.prepare_lm_data

stack_mod = importlib.import_module('08_stacking_layers')
Transformer = stack_mod.Transformer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trainer_creation():
    """TransformerTrainer can be instantiated with a Transformer model."""
    vocab_size = 50
    model = Transformer(
        vocab_size=vocab_size, embed_dim=32, num_heads=2, num_layers=1, seed=42
    )
    trainer = TransformerTrainer(model, lr=0.01)
    assert trainer is not None
    assert trainer.model is model
    assert trainer.lr == 0.01


def test_compute_loss_returns_positive():
    """compute_loss returns a positive scalar for any valid input."""
    vocab_size = 20
    model = Transformer(
        vocab_size=vocab_size, embed_dim=32, num_heads=2, num_layers=1, seed=42
    )
    trainer = TransformerTrainer(model, lr=0.01)
    loss = trainer.compute_loss([1, 5, 3], [5, 3, 2])
    assert loss > 0


def test_training_reduces_loss():
    """After multiple training steps on a fixed pattern, loss should decrease."""
    vocab_size = 10
    model = Transformer(
        vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=1, seed=42
    )
    trainer = TransformerTrainer(model, lr=0.05)

    # Simple repeating sequence
    input_seq = [1, 2, 3, 4, 5]
    target_seq = [2, 3, 4, 5, 6]

    loss1 = trainer.train_step(input_seq, target_seq)
    for _ in range(30):
        loss = trainer.train_step(input_seq, target_seq)

    assert loss < loss1, (
        f"Loss should decrease with training: initial={loss1:.4f}, final={loss:.4f}"
    )


def test_prepare_lm_data():
    """prepare_lm_data splits a sequence into (input[:-1], target[1:])."""
    inp, tgt = prepare_lm_data([1, 2, 3, 4, 5])
    assert inp == [1, 2, 3, 4]
    assert tgt == [2, 3, 4, 5]


def test_prepare_lm_data_short():
    """prepare_lm_data works with a minimal 2-token sequence."""
    inp, tgt = prepare_lm_data([7, 9])
    assert inp == [7]
    assert tgt == [9]


def test_train_method():
    """The train() method runs multiple epochs and returns loss history."""
    vocab_size = 10
    model = Transformer(
        vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=1, seed=42
    )
    trainer = TransformerTrainer(model, lr=0.05)

    dataset = [
        ([1, 2, 3], [2, 3, 4]),
        ([3, 4, 5], [4, 5, 6]),
    ]
    losses = trainer.train(dataset, epochs=5, verbose=False)

    assert len(losses) == 5
    assert all(l > 0 for l in losses), "All epoch losses should be positive"


def test_train_converges():
    """Training on a simple dataset for enough epochs should reduce loss."""
    vocab_size = 8
    model = Transformer(
        vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=1, seed=42
    )
    trainer = TransformerTrainer(model, lr=0.05)

    dataset = [([1, 2, 3, 4], [2, 3, 4, 5])]
    losses = trainer.train(dataset, epochs=20, verbose=False)

    assert losses[-1] < losses[0], (
        f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_get_hidden_states_shape():
    """Hidden states should have shape [seq_len x embed_dim]."""
    vocab_size = 10
    embed_dim = 16
    model = Transformer(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=2,
        num_layers=1, seed=42
    )
    trainer = TransformerTrainer(model, lr=0.01)

    tokens = [1, 2, 3]
    hidden = trainer._get_hidden_states(tokens)

    assert len(hidden) == 3
    assert len(hidden[0]) == embed_dim
