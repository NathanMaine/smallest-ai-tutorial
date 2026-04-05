"""
Tests for Chapter 4 — LSTM Cell with Forget, Input, and Output Gates.

Verifies:
  - Cell creation with correct dimensions
  - Initial states are zero
  - Step returns correct-length hidden state
  - Both hidden and cell state change after a step
  - Gate values are in the sigmoid range (0, 1)
  - Hidden values are in the tanh range [-1, 1]
  - Same input at different sequence positions produces different output (memory)
  - Reset returns both states to zeros
"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '01-mlp-from-scratch', 'solution'))

lstm_mod = importlib.import_module('04_lstm_cell')
LSTMCell = lstm_mod.LSTMCell


def test_lstm_cell_creation():
    """LSTMCell stores correct hidden_size and state lengths."""
    cell = LSTMCell(input_size=3, hidden_size=5, seed=42)
    assert cell.hidden_size == 5, f"Expected hidden_size=5, got {cell.hidden_size}"
    assert len(cell.hidden_state) == 5, f"hidden_state length {len(cell.hidden_state)}, expected 5"
    assert len(cell.cell_state) == 5, f"cell_state length {len(cell.cell_state)}, expected 5"


def test_lstm_cell_initial_states_zero():
    """Both hidden and cell state start as all zeros."""
    cell = LSTMCell(input_size=3, hidden_size=4, seed=42)
    assert all(v == 0.0 for v in cell.hidden_state), "hidden_state not all zeros"
    assert all(v == 0.0 for v in cell.cell_state), "cell_state not all zeros"


def test_lstm_step_returns_hidden():
    """step() returns a list of correct length."""
    cell = LSTMCell(input_size=2, hidden_size=4, seed=42)
    h = cell.step([1.0, 0.5])
    assert isinstance(h, list), f"Expected list, got {type(h)}"
    assert len(h) == 4, f"Expected length 4, got {len(h)}"


def test_lstm_step_updates_both_states():
    """After a non-zero input step, both hidden and cell state are non-zero."""
    cell = LSTMCell(input_size=2, hidden_size=4, seed=42)
    cell.step([1.0, -0.5])

    hidden_nonzero = any(abs(v) > 1e-10 for v in cell.hidden_state)
    cell_nonzero = any(abs(v) > 1e-10 for v in cell.cell_state)
    assert hidden_nonzero, "hidden_state is still all zeros after step"
    assert cell_nonzero, "cell_state is still all zeros after step"


def test_lstm_gate_values_in_range():
    """Forget, input, and output gates must all be in (0, 1) after a step."""
    cell = LSTMCell(input_size=3, hidden_size=5, seed=42)
    cell.step([1.0, 0.5, -0.3])

    for name, gate in [("forget", cell.forget_gate),
                       ("input", cell.input_gate),
                       ("output", cell.output_gate)]:
        for i, v in enumerate(gate):
            assert 0.0 < v < 1.0, f"{name}_gate[{i}] = {v} not in (0, 1)"


def test_lstm_hidden_in_tanh_range():
    """Hidden state values must be in [-1, 1] (product of sigmoid and tanh)."""
    cell = LSTMCell(input_size=2, hidden_size=4, seed=42)
    cell.step([1.0, 0.5])

    for i, v in enumerate(cell.hidden_state):
        assert -1.0 <= v <= 1.0, f"hidden[{i}] = {v} not in [-1, 1]"


def test_lstm_sequence_memory():
    """Same input at different positions must produce different hidden states."""
    cell = LSTMCell(input_size=2, hidden_size=4, seed=42)

    # First step with [1, 0]
    h1 = cell.step([1.0, 0.0])

    # Give a different input, then the same [1, 0] again
    cell.step([0.0, 1.0])
    h2 = cell.step([1.0, 0.0])

    # h1 and h2 should differ because the cell has memory of the middle step
    diffs = [abs(a - b) for a, b in zip(h1, h2)]
    assert max(diffs) > 1e-6, "Same input at different positions produced identical output — no memory"


def test_lstm_reset():
    """reset() returns both hidden and cell state to zeros."""
    cell = LSTMCell(input_size=2, hidden_size=4, seed=42)

    # Run a few steps to populate state
    cell.step([1.0, 0.5])
    cell.step([0.0, -1.0])

    # Verify states are non-zero before reset
    assert any(abs(v) > 1e-10 for v in cell.hidden_state), "States should be non-zero before reset"

    cell.reset()

    assert all(v == 0.0 for v in cell.hidden_state), "hidden_state not zero after reset"
    assert all(v == 0.0 for v in cell.cell_state), "cell_state not zero after reset"


if __name__ == "__main__":
    tests = [
        test_lstm_cell_creation,
        test_lstm_cell_initial_states_zero,
        test_lstm_step_returns_hidden,
        test_lstm_step_updates_both_states,
        test_lstm_gate_values_in_range,
        test_lstm_hidden_in_tanh_range,
        test_lstm_sequence_memory,
        test_lstm_reset,
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
