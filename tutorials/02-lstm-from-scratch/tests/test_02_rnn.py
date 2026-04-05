"""
Tests for Chapter 2 — Vanilla RNN with BPTT.

Verifies:
  - RNNCell forward produces correct-size hidden states in [-1, 1]
  - Different hidden states produce different outputs (memory matters)
  - Full RNN forward processes variable-length sequences
  - Outputs differ at each time step (recurrence has effect)
  - BPTT backward produces non-zero gradients
  - Training reduces loss over multiple steps
"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '01-mlp-from-scratch', 'solution'))

rnn_mod = importlib.import_module('02_vanilla_rnn')
RNNCell = rnn_mod.RNNCell
RNN = rnn_mod.RNN

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax


def test_rnn_cell_forward():
    """RNNCell forward with zero hidden state produces valid output."""
    cell = RNNCell(3, 4)
    h = cell.forward([1.0, 0.5, -0.3], [0.0] * 4)
    assert len(h) == 4, f"Expected hidden size 4, got {len(h)}"
    for i, v in enumerate(h):
        assert -1.0 <= v <= 1.0, f"h[{i}] = {v} not in [-1, 1]"


def test_rnn_cell_different_hidden_different_output():
    """Same input with different hidden states must produce different outputs."""
    cell = RNNCell(3, 4, seed=42)
    x = [1.0, 0.5, -0.3]

    h1 = cell.forward(x, [0.0] * 4)
    h2 = cell.forward(x, [0.5, -0.5, 0.3, -0.3])

    # At least one element should differ
    diffs = [abs(a - b) for a, b in zip(h1, h2)]
    assert max(diffs) > 1e-6, "Same input with different hidden states should produce different output"


def test_rnn_forward_sequence():
    """Full RNN forward processes a 3-step sequence with correct shapes."""
    rnn = RNN(2, 4, 3)
    sequence = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    outputs, hiddens = rnn.forward(sequence)

    # 3 outputs, each of size 3 (output_size)
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
    for t, out in enumerate(outputs):
        assert len(out) == 3, f"Output at t={t} has length {len(out)}, expected 3"

    # 3 hidden states, each of size 4 (hidden_size)
    assert len(hiddens) == 3, f"Expected 3 hidden states, got {len(hiddens)}"
    for t, h in enumerate(hiddens):
        assert len(h) == 4, f"Hidden at t={t} has length {len(h)}, expected 4"


def test_rnn_output_changes_per_step():
    """Outputs should differ at each time step due to recurrence."""
    rnn = RNN(2, 4, 3, seed=42)
    sequence = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    outputs, _ = rnn.forward(sequence)

    # Check that not all outputs are identical
    all_same = True
    for t in range(1, len(outputs)):
        diffs = [abs(outputs[t][i] - outputs[0][i]) for i in range(len(outputs[0]))]
        if max(diffs) > 1e-6:
            all_same = False
            break
    assert not all_same, "All outputs are identical — recurrence has no effect"


def test_rnn_backward_produces_gradients():
    """After backward, gradient accumulators should be non-zero."""
    rnn = RNN(2, 4, 3, seed=42)
    sequence = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    targets = [0, 1, 2]

    rnn._zero_all_gradients()
    rnn.forward(sequence)
    rnn.backward(targets)

    # Check that at least some gradients are non-zero
    def matrix_max_abs(m):
        return max(abs(v) for row in m for v in row)

    assert matrix_max_abs(rnn.cell.dW_input) > 1e-10, "dW_input is all zeros"
    assert matrix_max_abs(rnn.cell.dW_hidden) > 1e-10, "dW_hidden is all zeros"
    assert matrix_max_abs(rnn.dW_output) > 1e-10, "dW_output is all zeros"


def test_rnn_training_reduces_loss():
    """After 20 train_steps, loss should decrease."""
    rnn = RNN(2, 4, 3, seed=42)
    sequence = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    targets = [0, 1, 2]

    initial_loss = rnn.train_step(sequence, targets, lr=0.1)

    # Train for 19 more steps (20 total)
    for _ in range(19):
        final_loss = rnn.train_step(sequence, targets, lr=0.1)

    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


if __name__ == "__main__":
    tests = [
        test_rnn_cell_forward,
        test_rnn_cell_different_hidden_different_output,
        test_rnn_forward_sequence,
        test_rnn_output_changes_per_step,
        test_rnn_backward_produces_gradients,
        test_rnn_training_reduces_loss,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
