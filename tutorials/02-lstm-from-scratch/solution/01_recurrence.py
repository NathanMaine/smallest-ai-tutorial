"""
Chapter 1 — Recurrence: Hidden State and Memory
================================================

What this module teaches
-------------------------
This is the first chapter of Level B (Phonics). We have learned how to build
a feedforward MLP that maps a fixed-size input to a fixed-size output. But
MLPs are stateless: every forward pass is independent. Feed the same vector
twice and you get the same output, always.

Language, speech, music, and time-series data are fundamentally sequential.
The meaning of a word depends on what came before it. The sentiment of a
sentence cannot be decided from a single token. To handle sequences we need
networks that carry **hidden state** — a compressed memory of everything seen
so far.

Why MLPs can't handle sequences
---------------------------------
An MLP processes one fixed-length vector at a time. To handle a sequence of
length T with an MLP you would have to concatenate all T inputs, fixing the
sequence length forever. That fails for variable-length sequences, wastes
parameters on early positions, and learns no notion of "what comes before
what." MLPs have no temporal awareness.

The recurrent solution
-----------------------
A Recurrent Neural Network (RNN) maintains a hidden state vector h that is
updated at every time step:

    h_t = tanh(W_input · x_t  +  W_hidden · h_{t-1}  +  b)

At each step:
  - x_t  is the current input vector
  - h_{t-1} is the memory of everything seen before step t
  - W_input maps the new input into hidden space
  - W_hidden maps yesterday's memory into today's memory
  - tanh squashes the result to (-1, 1), preventing explosion

The critical insight: the same input x = [1, 0] produces a *different* output
depending on h_{t-1}. Position 1 of a sequence and position 5 are treated
differently — exactly what we need for language.

The tanh activation
--------------------
We choose tanh (hyperbolic tangent) rather than sigmoid because:
  - Output range is (-1, 1), centered at zero (sigmoid is (0, 1))
  - Zero-centered outputs reduce the "all gradients same sign" problem
  - The derivative is 1 - tanh²(z), convenient to compute from the output
  - Widely used in classical RNNs (LSTM cells also use tanh internally)

Chapter roadmap
---------------
  Chapter 1 (this file): Recurrence — hidden state, SimpleMemoryCell
  Chapter 2:             Sequences — variable-length input, embeddings
  Chapter 3:             LSTM — gated memory, forget/input/output gates
  Chapter 4:             Attention — weighting past states by relevance
"""

import importlib
import sys
import os
import math
import random

# ---------------------------------------------------------------------------
# Import shared math primitives from Level A
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product
vector_add = math_fn.vector_add


# ---------------------------------------------------------------------------
# Activation: tanh
# ---------------------------------------------------------------------------

def tanh(z):
    """
    Hyperbolic tangent activation function.

    Outputs (-1, 1), centered at zero — the standard RNN activation.

    The input is clamped to [-500, 500] before evaluation to prevent
    overflow in math.cosh (which grows as e^|z|). In practice, well-trained
    networks never produce inputs outside [-10, 10], so clamping is a
    safety measure, not a distortion.

    Parameters
    ----------
    z : float — pre-activation value (any real number)

    Returns
    -------
    float — tanh(z) in the open interval (-1, 1)

    Examples
    --------
    >>> round(tanh(0.0), 6)
    0.0
    >>> round(tanh(1.0), 6)
    0.761594
    >>> tanh(1000)  # large positive → saturates near 1
    1.0
    >>> tanh(-1000)  # large negative → saturates near -1
    -1.0
    """
    z = max(-500.0, min(500.0, z))
    return math.tanh(z)


def tanh_derivative(a):
    """
    Derivative of tanh, given the tanh *output* (not the pre-activation).

    If a = tanh(z), then  d(tanh)/dz = 1 - a².

    This form is convenient because during backpropagation we already have
    the activations stored from the forward pass — we do not need to
    re-compute tanh(z) again.

    Parameters
    ----------
    a : float — the tanh activation output, i.e. tanh(z) for some z.
                 Should be in (-1, 1).

    Returns
    -------
    float — gradient of tanh at the point whose tanh value is a.

    Examples
    --------
    >>> round(tanh_derivative(0.0), 6)
    1.0
    >>> round(tanh_derivative(1.0), 6)  # saturated → near-zero gradient
    0.0
    """
    return 1.0 - a * a


# ---------------------------------------------------------------------------
# SimpleMemoryCell
# ---------------------------------------------------------------------------

class SimpleMemoryCell:
    """
    The simplest possible recurrent unit: one tanh cell with hidden state.

    Architecture
    ------------
    At each time step t, given input x_t and previous hidden state h_{t-1}:

        input_contrib  = W_input  @ x_t          (shape: hidden_size)
        hidden_contrib = W_hidden @ h_{t-1}       (shape: hidden_size)
        combined       = input_contrib + hidden_contrib + bias
        h_t            = tanh(combined)            (shape: hidden_size)

    The cell returns h_t and stores it for the next step.

    Parameters
    ----------
    input_size  : int — dimensionality of each input vector x_t
    hidden_size : int — dimensionality of the hidden state h_t
    seed        : int — seed for the local RNG (reproducible init)

    Weight initialisation
    ---------------------
    Xavier (Glorot) uniform initialisation:

        limit = sqrt(6 / (fan_in + fan_out))
        W ~ Uniform(-limit, limit)

    For W_input:  fan_in = input_size,  fan_out = hidden_size
    For W_hidden: fan_in = hidden_size, fan_out = hidden_size
    Bias is initialised to zero (standard practice).

    The RNG is a *local* random.Random instance seeded with `seed`.
    This avoids polluting the global random state.
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Local RNG — does NOT call random.seed() so global state is untouched
        rng = random.Random(seed)

        # Xavier initialisation limits
        limit_input = math.sqrt(6.0 / (input_size + hidden_size))
        limit_hidden = math.sqrt(6.0 / (hidden_size + hidden_size))

        # W_input: shape [hidden_size x input_size]
        self.W_input = [
            [rng.uniform(-limit_input, limit_input) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]

        # W_hidden: shape [hidden_size x hidden_size]
        self.W_hidden = [
            [rng.uniform(-limit_hidden, limit_hidden) for _ in range(hidden_size)]
            for _ in range(hidden_size)
        ]

        # Bias: shape [hidden_size], initialised to zero
        self.bias = [0.0] * hidden_size

        # Hidden state: shape [hidden_size], starts at zero
        self.hidden_state = [0.0] * hidden_size

    def step(self, input_vec):
        """
        Process one time step: update and return the hidden state.

        Computes:
            h_t = tanh(W_input @ x_t + W_hidden @ h_{t-1} + bias)

        The internal hidden_state is updated in-place. The returned list is
        a copy, so the caller's reference is stable across multiple steps.

        Parameters
        ----------
        input_vec : list of float, length = input_size

        Returns
        -------
        list of float — the new hidden state h_t, length = hidden_size
        """
        # Linear projections
        input_contrib = [dot_product(row, input_vec) for row in self.W_input]
        hidden_contrib = [dot_product(row, self.hidden_state) for row in self.W_hidden]

        # Combine: input + recurrent + bias
        combined = vector_add(vector_add(input_contrib, hidden_contrib), self.bias)

        # Apply tanh element-wise and update hidden state
        self.hidden_state = [tanh(z) for z in combined]

        return list(self.hidden_state)  # return a copy

    def reset(self):
        """
        Zero the hidden state, as if the cell has never seen any input.

        Call this between independent sequences (e.g., between training
        examples) so that sequence A's memory does not bleed into sequence B.
        """
        self.hidden_state = [0.0] * self.hidden_size

    def process_sequence(self, sequence):
        """
        Process a full sequence of input vectors from scratch.

        Resets hidden state first (fresh memory), then steps through every
        element of the sequence in order, collecting each hidden state.

        Parameters
        ----------
        sequence : list of list of float
            Each element is one time step's input vector (length = input_size).

        Returns
        -------
        list of list of float
            One hidden state per time step, in order. Length = len(sequence).
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
    print("=" * 60)
    print("Chapter 1 — Recurrence: Hidden State and Memory")
    print("=" * 60)

    # --- Demo 1: Hidden state evolving through a sequence ---
    print("\n[Demo 1] Hidden state evolution through a short sequence")
    print("-" * 60)

    cell = SimpleMemoryCell(input_size=2, hidden_size=4, seed=42)
    sequence = [
        [1.0, 0.0],   # step 1: first token
        [0.0, 1.0],   # step 2: second token
        [1.0, 1.0],   # step 3: third token
        [0.5, -0.5],  # step 4: fourth token
    ]

    print(f"  initial hidden state: {[round(v, 4) for v in cell.hidden_state]}")
    for t, x_t in enumerate(sequence, start=1):
        h_t = cell.step(x_t)
        print(f"  step {t}  input={x_t}  →  h={[round(v, 4) for v in h_t]}")

    # --- Demo 2: Memory demonstration ---
    print("\n[Demo 2] Same input, different history → different output")
    print("-" * 60)
    print("  This is the key property that makes RNNs useful for sequences.")

    cell2 = SimpleMemoryCell(input_size=2, hidden_size=4, seed=42)

    # Fresh cell: first input is [1, 0]
    output_fresh = cell2.step([1.0, 0.0])
    print(f"  Fresh cell,  input=[1, 0] → {[round(v, 4) for v in output_fresh]}")

    # Reset and build up history first
    cell2.reset()
    cell2.step([0.0, 1.0])
    cell2.step([0.0, 1.0])
    output_history = cell2.step([1.0, 0.0])
    print(f"  After [0,1],[0,1], input=[1, 0] → {[round(v, 4) for v in output_history]}")

    same = output_fresh == output_history
    print(f"\n  Outputs identical? {same}  (expected: False)")
    print("  The cell remembers its history — recurrence gives it memory.")

    # --- Demo 3: tanh properties ---
    print("\n[Demo 3] tanh activation properties")
    print("-" * 60)
    test_vals = [-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0]
    for z in test_vals:
        a = tanh(z)
        da = tanh_derivative(a)
        print(f"  tanh({z:6.1f}) = {a:+.6f}   derivative = {da:.6f}")

    print("\n[Done] SimpleMemoryCell working correctly.")
    print("Next: Chapter 2 — Sequences: variable-length input and embeddings.")
