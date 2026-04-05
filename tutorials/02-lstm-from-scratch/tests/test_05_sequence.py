"""
Tests for Chapter 5 — Trainable LSTM Sequence Model with BPTT.

Verifies:
  - Model creation with correct dimensions
  - Forward pass output shape
  - Predict returns valid class indices
  - Training reduces loss over multiple steps
"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '01-mlp-from-scratch', 'solution'))

seq_mod = importlib.import_module('05_sequence_model')
LSTMSequenceModel = seq_mod.LSTMSequenceModel


def test_model_creation():
    """LSTMSequenceModel can be instantiated with correct parameters."""
    model = LSTMSequenceModel(input_size=5, hidden_size=8, output_size=3, seed=42)
    assert model is not None


def test_model_forward_output_shape():
    """Forward pass returns one output per time step with correct dimensionality."""
    model = LSTMSequenceModel(input_size=5, hidden_size=8, output_size=3, seed=42)
    outputs = model.forward([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
    assert len(outputs[0]) == 3, f"Expected output_size=3, got {len(outputs[0])}"


def test_model_predict_returns_indices():
    """Predict returns integer class indices in valid range."""
    model = LSTMSequenceModel(input_size=5, hidden_size=8, output_size=3, seed=42)
    predictions = model.predict([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    assert len(predictions) == 2, f"Expected 2 predictions, got {len(predictions)}"
    for p in predictions:
        assert 0 <= p < 3, f"Prediction {p} out of range [0, 3)"


def test_model_training_reduces_loss():
    """Loss decreases after repeated training steps on the same data."""
    model = LSTMSequenceModel(input_size=3, hidden_size=8, output_size=3, seed=42)
    sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    targets = [0, 1, 2]
    loss1 = model.train_step(sequence, targets, lr=0.05)
    for _ in range(50):
        loss = model.train_step(sequence, targets, lr=0.05)
    assert loss < loss1, f"Loss did not decrease: initial={loss1:.6f}, final={loss:.6f}"
