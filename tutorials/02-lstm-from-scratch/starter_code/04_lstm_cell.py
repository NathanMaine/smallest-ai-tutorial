"""
Chapter 4 — The LSTM Revolution: Gated Memory
===============================================

What this module teaches
-------------------------
The LSTM solves the vanishing gradient problem with THREE learned gates:

  1. Forget gate (f_t)  — decides what to ERASE from cell state (sigmoid output: 0=forget, 1=keep)
  2. Input gate (i_t)   — decides what NEW information to WRITE
  3. Output gate (o_t)  — decides what to READ from cell state into hidden output

LSTM equations at each time step
----------------------------------
    combined = concat(x_t, h_{t-1})          # [input_size + hidden_size]

    f_t = sigmoid(W_f @ combined + b_f)       # Forget gate
    i_t = sigmoid(W_i @ combined + b_i)       # Input gate
    o_t = sigmoid(W_o @ combined + b_o)       # Output gate
    c_hat = tanh(W_c @ combined + b_c)        # Candidate cell content

    c_t = f_t * c_{t-1} + i_t * c_hat        # New cell state
    h_t = o_t * tanh(c_t)                     # New hidden state

Forget bias = 1.0 (not 0.0)
-------------------------------
Standard practice: forget gate bias starts at 1.0 so the network begins by
remembering everything, then learns what to forget. Starting at 0 makes
early training much harder.

Builds on: 01_recurrence.py (tanh), 01-mlp-from-scratch (sigmoid, dot_product)
"""

import importlib
import sys
import os
import math
import random

# Import from previous chapters
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

math_fn    = importlib.import_module('01_math_foundations')
neuron_mod = importlib.import_module('02_single_neuron')
rec_mod    = importlib.import_module('01_recurrence')

dot_product = math_fn.dot_product
vector_add  = math_fn.vector_add
sigmoid     = neuron_mod.sigmoid
tanh        = rec_mod.tanh


def elementwise_mul(a, b):
    """Hadamard (element-wise) product of two equal-length vectors."""
    return [ai * bi for ai, bi in zip(a, b)]


class LSTMCell:
    """
    Long Short-Term Memory cell built from scratch.

    All matrix operations use dot products from Tutorial 01. No NumPy.

    Parameters
    ----------
    input_size  : int — dimensionality of input vectors
    hidden_size : int — dimensionality of hidden and cell states
    seed        : int — seed for reproducible Xavier initialisation
    """

    def __init__(self, input_size, hidden_size, seed=42):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        combined_size    = input_size + hidden_size

        rng   = random.Random(seed)
        limit = math.sqrt(6.0 / (combined_size + hidden_size))

        def make_weight_matrix():
            return [
                [rng.uniform(-limit, limit) for _ in range(combined_size)]
                for _ in range(hidden_size)
            ]

        # Gate weight matrices: each [hidden_size x combined_size]
        self.W_f = make_weight_matrix()  # Forget gate
        self.W_i = make_weight_matrix()  # Input gate
        self.W_o = make_weight_matrix()  # Output gate
        self.W_c = make_weight_matrix()  # Candidate

        # Biases: each [hidden_size]
        # NOTE: forget bias starts at 1.0 (not 0.0) — important!
        self.b_f = [1.0] * hidden_size
        self.b_i = [0.0] * hidden_size
        self.b_o = [0.0] * hidden_size
        self.b_c = [0.0] * hidden_size

        # States initialised to zero
        self.hidden_state = [0.0] * hidden_size
        self.cell_state   = [0.0] * hidden_size

        # Gate values stored for inspection
        self.forget_gate = [0.0] * hidden_size
        self.input_gate  = [0.0] * hidden_size
        self.output_gate = [0.0] * hidden_size
        self.candidate   = [0.0] * hidden_size

    def step(self, input_vec):
        """
        Process one time step through the LSTM cell.

        Steps:
          1. Concatenate input_vec with self.hidden_state → combined
          2. Compute forget gate:  f = sigmoid(W_f @ combined + b_f)
          3. Compute input gate:   i = sigmoid(W_i @ combined + b_i)
          4. Compute output gate:  o = sigmoid(W_o @ combined + b_o)
          5. Compute candidate:    c_hat = tanh(W_c @ combined + b_c)
          6. Update cell state:    c_t = f * c_{t-1} + i * c_hat
          7. Update hidden state:  h_t = o * tanh(c_t)

        Store the gate values in self.forget_gate, self.input_gate,
        self.output_gate, self.candidate for inspection.

        Parameters
        ----------
        input_vec : list of float, length = input_size

        Returns
        -------
        list of float — copy of the new hidden state, length = hidden_size
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  1. combined = list(input_vec) + list(self.hidden_state)\n"
            "  2. self.forget_gate = [sigmoid(dot_product(self.W_f[j], combined) + self.b_f[j]) for j in range(self.hidden_size)]\n"
            "     (same pattern for input, output, candidate — use tanh for candidate)\n"
            "  3. self.cell_state = vector_add(elementwise_mul(self.forget_gate, self.cell_state),\n"
            "                                  elementwise_mul(self.input_gate, self.candidate))\n"
            "  4. self.hidden_state = elementwise_mul(self.output_gate, [tanh(c) for c in self.cell_state])\n"
            "  5. return list(self.hidden_state)"
        )

    def reset(self):
        """Zero both hidden and cell state."""
        self.hidden_state = [0.0] * self.hidden_size
        self.cell_state   = [0.0] * self.hidden_size

    def process_sequence(self, sequence):
        """Process a full sequence from scratch, return all hidden states."""
        self.reset()
        return [self.step(x_t) for x_t in sequence]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lstm = LSTMCell(input_size=2, hidden_size=3, seed=42)

    # Show gates after each step
    sequence = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    for t, x in enumerate(sequence, 1):
        h = lstm.step(x)
        print(f"Step {t}: f={[round(v,3) for v in lstm.forget_gate]}  "
              f"i={[round(v,3) for v in lstm.input_gate]}  "
              f"h={[round(v,3) for v in h]}")

    print("\nForget gate values close to 1.0 = the cell is remembering.")
    print("Input gate values close to 0.0  = the cell is not writing much new info.")
