"""
Chapter 4 — The LSTM Revolution: Gated Memory
===============================================

What this module teaches
-------------------------
The Long Short-Term Memory (LSTM) cell solves the vanishing gradient problem
that cripples vanilla RNNs. Where a vanilla RNN has a single tanh gate that
overwrites hidden state at every step, the LSTM uses THREE learned gates to
control information flow:

  1. Forget gate (f_t)  — decides what to ERASE from cell state (0=forget, 1=keep)
  2. Input gate (i_t)   — decides what NEW information to WRITE to cell state
  3. Output gate (o_t)  — decides what to READ from cell state into hidden output

The cell state is the "highway" for gradients
-----------------------------------------------
The key innovation is the cell state c_t, which flows through time with only
element-wise operations (multiply + add). Gradients can travel along this
highway for hundreds of steps without vanishing or exploding. The gates learn
to open and close, giving the network fine-grained control over its memory.

LSTM equations at each time step
----------------------------------
    combined = concat(x_t, h_{t-1})      # [input_size + hidden_size]

    f_t = sigmoid(W_f @ combined + b_f)   # Forget gate
    i_t = sigmoid(W_i @ combined + b_i)   # Input gate
    o_t = sigmoid(W_o @ combined + b_o)   # Output gate
    c_hat = tanh(W_c @ combined + b_c)    # Candidate cell content

    c_t = f_t * c_{t-1} + i_t * c_hat    # New cell state
    h_t = o_t * tanh(c_t)                 # New hidden state (output)

Forget bias = 1.0
-------------------
The forget gate bias is initialized to 1.0 (not 0.0). This is standard
practice (Jozefowicz et al., 2015): the network starts by remembering
everything, and learns what to forget during training. Without this, the
forget gate starts at sigmoid(0) = 0.5, discarding half the cell state
from the very first step.

Chapter roadmap
----------------
  Chapter 1: Recurrence — hidden state, SimpleMemoryCell
  Chapter 2: Vanilla RNN — forward through time and BPTT
  Chapter 3: (reserved)
  Chapter 4 (this file): LSTM — gated memory, forget/input/output gates
"""

import importlib
import sys
import os
import math
import random

# ---------------------------------------------------------------------------
# Import shared primitives from Level A and Level B ch1
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(__file__))

math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product
vector_add = math_fn.vector_add

neuron_mod = importlib.import_module('02_single_neuron')
sigmoid = neuron_mod.sigmoid

recurrence_mod = importlib.import_module('01_recurrence')
tanh = recurrence_mod.tanh


# ---------------------------------------------------------------------------
# Helper: element-wise multiplication of two vectors
# ---------------------------------------------------------------------------

def elementwise_mul(a, b):
    """Hadamard (element-wise) product of two equal-length vectors."""
    return [ai * bi for ai, bi in zip(a, b)]


# ---------------------------------------------------------------------------
# LSTMCell — a single LSTM cell with forget, input, and output gates
# ---------------------------------------------------------------------------

class LSTMCell:
    """
    Long Short-Term Memory cell built from scratch.

    Three sigmoid gates control information flow through the cell state.
    All matrix operations use dot products from Level A. No NumPy.

    Parameters
    ----------
    input_size  : int — dimensionality of input vectors
    hidden_size : int — dimensionality of hidden state and cell state
    seed        : int — seed for reproducible Xavier initialisation
    """

    def __init__(self, input_size, hidden_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        rng = random.Random(seed)

        # Xavier (Glorot) uniform initialisation
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
        # IMPORTANT: forget bias starts at 1.0 so the network remembers by default
        self.b_f = [1.0] * hidden_size
        self.b_i = [0.0] * hidden_size
        self.b_o = [0.0] * hidden_size
        self.b_c = [0.0] * hidden_size

        # States: both [hidden_size], initialised to zero
        self.hidden_state = [0.0] * hidden_size
        self.cell_state = [0.0] * hidden_size

        # Gate values stored for inspection (after each step)
        self.forget_gate = [0.0] * hidden_size
        self.input_gate = [0.0] * hidden_size
        self.output_gate = [0.0] * hidden_size
        self.candidate = [0.0] * hidden_size

    def step(self, input_vec):
        """
        Process one time step through the LSTM cell.

        Concatenates input with previous hidden state, computes all three
        gates and the candidate, updates cell state and hidden state.

        Parameters
        ----------
        input_vec : list of float, length = input_size

        Returns
        -------
        list of float — copy of the new hidden state, length = hidden_size
        """
        # 1. Concatenate input and previous hidden state
        combined = list(input_vec) + list(self.hidden_state)

        # 2. Compute gates and candidate via dot products + activations
        self.forget_gate = [
            sigmoid(dot_product(self.W_f[j], combined) + self.b_f[j])
            for j in range(self.hidden_size)
        ]
        self.input_gate = [
            sigmoid(dot_product(self.W_i[j], combined) + self.b_i[j])
            for j in range(self.hidden_size)
        ]
        self.output_gate = [
            sigmoid(dot_product(self.W_o[j], combined) + self.b_o[j])
            for j in range(self.hidden_size)
        ]
        self.candidate = [
            tanh(dot_product(self.W_c[j], combined) + self.b_c[j])
            for j in range(self.hidden_size)
        ]

        # 3. Update cell state: c_t = f_t * c_{t-1} + i_t * candidate
        self.cell_state = vector_add(
            elementwise_mul(self.forget_gate, self.cell_state),
            elementwise_mul(self.input_gate, self.candidate),
        )

        # 4. Update hidden state: h_t = o_t * tanh(c_t)
        self.hidden_state = elementwise_mul(
            self.output_gate,
            [tanh(c) for c in self.cell_state],
        )

        return list(self.hidden_state)  # return a copy

    def reset(self):
        """Zero both hidden state and cell state, as if no input was ever seen."""
        self.hidden_state = [0.0] * self.hidden_size
        self.cell_state = [0.0] * self.hidden_size

    def process_sequence(self, sequence):
        """
        Process a full sequence from scratch.

        Resets both states, steps through every input, returns all hidden states.

        Parameters
        ----------
        sequence : list of list of float
            Each element is one time step's input vector (length = input_size).

        Returns
        -------
        list of list of float
            One hidden state per time step.
        """
        self.reset()
        states = []
        for x_t in sequence:
            h_t = self.step(x_t)
            states.append(h_t)
        return states


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 4 — The LSTM Revolution: Gated Memory")
    print("=" * 65)

    # --- Demo 1: Gate values at each step ---
    print("\n[Demo 1] LSTM gate values through a 4-step sequence")
    print("-" * 65)

    lstm = LSTMCell(input_size=2, hidden_size=3, seed=42)
    sequence = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, -0.5],
    ]

    print(f"  Initial hidden: {[round(v, 4) for v in lstm.hidden_state]}")
    print(f"  Initial cell:   {[round(v, 4) for v in lstm.cell_state]}")
    print()

    for t, x_t in enumerate(sequence, start=1):
        h_t = lstm.step(x_t)
        print(f"  Step {t}  input={x_t}")
        print(f"    forget gate: {[round(v, 4) for v in lstm.forget_gate]}")
        print(f"    input gate:  {[round(v, 4) for v in lstm.input_gate]}")
        print(f"    output gate: {[round(v, 4) for v in lstm.output_gate]}")
        print(f"    candidate:   {[round(v, 4) for v in lstm.candidate]}")
        print(f"    cell state:  {[round(v, 4) for v in lstm.cell_state]}")
        print(f"    hidden:      {[round(v, 4) for v in h_t]}")
        print()

    # --- Demo 2: Compare LSTM memory vs vanilla RNN ---
    print("[Demo 2] LSTM vs Vanilla RNN — memory persistence")
    print("-" * 65)

    # Import vanilla RNN's SimpleMemoryCell for comparison
    SimpleMemoryCell = recurrence_mod.SimpleMemoryCell

    rnn_cell = SimpleMemoryCell(input_size=2, hidden_size=3, seed=42)
    lstm_cell = LSTMCell(input_size=2, hidden_size=3, seed=42)

    # Feed a "signal" then several blank inputs
    signal = [1.0, 1.0]
    blanks = [[0.0, 0.0]] * 6

    full_seq = [signal] + blanks

    rnn_states = rnn_cell.process_sequence(full_seq)
    lstm_states = lstm_cell.process_sequence(full_seq)

    print("  After [1,1] signal followed by 6 blank [0,0] inputs:")
    print(f"  {'Step':<6} {'RNN h norm':>12} {'LSTM h norm':>12}")
    for t, (rh, lh) in enumerate(zip(rnn_states, lstm_states)):
        rnn_norm = math.sqrt(sum(v**2 for v in rh))
        lstm_norm = math.sqrt(sum(v**2 for v in lh))
        label = " <-- signal" if t == 0 else ""
        print(f"  {t:<6} {rnn_norm:>12.6f} {lstm_norm:>12.6f}{label}")

    print()
    print("  The LSTM retains the signal much longer because its forget gate")
    print("  keeps cell state alive. The vanilla RNN's hidden state decays")
    print("  exponentially through the tanh squashing at each step.")
    print()
    print("=" * 65)
    print("Chapter 4 complete.")
