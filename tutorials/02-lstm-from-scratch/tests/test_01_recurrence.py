"""
Tests for Chapter 1, Level B — Recurrence: Hidden State and Memory

Validates SimpleMemoryCell behavior: initialization, stepping, resetting,
sequence processing, and the core property that the same input produces
different outputs depending on history (i.e., that hidden state matters).
"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '01-mlp-from-scratch', 'solution'))

recurrence = importlib.import_module('01_recurrence')
SimpleMemoryCell = recurrence.SimpleMemoryCell


def test_memory_cell_creation():
    """Cell reports the correct hidden_size after construction."""
    cell = SimpleMemoryCell(input_size=2, hidden_size=4, seed=42)
    assert len(cell.hidden_state) == 4, (
        f"Expected hidden_size=4, got {len(cell.hidden_state)}"
    )


def test_memory_cell_initial_hidden_zeros():
    """Hidden state is all zeros before any input is processed."""
    cell = SimpleMemoryCell(input_size=3, hidden_size=5, seed=0)
    assert all(v == 0.0 for v in cell.hidden_state), (
        f"Expected all-zero initial hidden state, got {cell.hidden_state}"
    )


def test_memory_cell_step_changes_hidden():
    """Calling step() with a non-trivial input changes the hidden state."""
    cell = SimpleMemoryCell(input_size=2, hidden_size=3, seed=42)
    original = list(cell.hidden_state)
    cell.step([1.0, 0.0])
    assert cell.hidden_state != original, (
        "Hidden state should change after a step, but it did not."
    )


def test_memory_cell_step_returns_output():
    """step() returns a list of length hidden_size."""
    cell = SimpleMemoryCell(input_size=2, hidden_size=4, seed=42)
    output = cell.step([0.5, -0.5])
    assert len(output) == 4, (
        f"Expected output length 4, got {len(output)}"
    )


def test_memory_cell_sequence_different_states():
    """Each step through [1,0], [0,1], [1,1] produces a distinct hidden state."""
    cell = SimpleMemoryCell(input_size=2, hidden_size=4, seed=7)
    sequence = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    states = cell.process_sequence(sequence)

    assert len(states) == 3, f"Expected 3 states, got {len(states)}"

    # All three states should be distinct (different history → different state)
    assert states[0] != states[1], "State after step 1 and step 2 should differ."
    assert states[1] != states[2], "State after step 2 and step 3 should differ."
    assert states[0] != states[2], "State after step 1 and step 3 should differ."


def test_memory_cell_reset():
    """After reset(), hidden state returns to all zeros."""
    cell = SimpleMemoryCell(input_size=2, hidden_size=3, seed=42)
    cell.step([1.0, 1.0])
    cell.step([0.5, -1.0])
    cell.reset()
    assert all(v == 0.0 for v in cell.hidden_state), (
        f"Expected zeros after reset, got {cell.hidden_state}"
    )


def test_memory_cell_same_input_different_history():
    """The same input vector produces a different output depending on history.

    Step 1: feed [1, 0] to a fresh cell  → output_a
    Steps 1-2: feed [0, 1] then [1, 0]  → output_b

    output_a and output_b must differ because the hidden state carries memory
    of what came before.
    """
    cell = SimpleMemoryCell(input_size=2, hidden_size=4, seed=99)

    # Fresh cell: first input is [1, 0]
    output_a = cell.step([1.0, 0.0])

    # Reset and build up history before presenting [1, 0]
    cell.reset()
    cell.step([0.0, 1.0])   # step 1: different history
    cell.step([0.0, 1.0])   # step 2: more history
    output_b = cell.step([1.0, 0.0])   # step 3: same input, different context

    assert output_a != output_b, (
        "Same input at different sequence positions should produce different "
        f"outputs.\noutput_a={output_a}\noutput_b={output_b}"
    )
