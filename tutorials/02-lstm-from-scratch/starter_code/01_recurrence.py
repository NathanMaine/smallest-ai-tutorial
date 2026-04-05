"""
Chapter 1 — Recurrence: Hidden State and Memory
================================================

What this module teaches
-------------------------
An MLP processes one input at a time with no memory of what came before.
Language, speech, and time-series data are fundamentally sequential — the
meaning of one token depends on what preceded it.

A Recurrent Neural Network (RNN) solves this with a *hidden state*: a vector
that summarises everything the network has seen so far. At each time step:

    h_t = tanh(W_input @ x_t  +  W_hidden @ h_{t-1}  +  bias)

Same input, different history → different output. That's the key property.

The tanh activation
--------------------
We use tanh rather than sigmoid because:
  - Output range (-1, 1) is zero-centered (sigmoid is (0, 1))
  - Derivative is 1 - tanh²(z), easy to compute from the output
  - Standard in classical RNNs and LSTM cells

Builds on: 01-mlp-from-scratch/solution (dot_product, vector_add)
"""

import importlib
import sys
import os
import math
import random

# Import shared math primitives from Tutorial 01
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product
vector_add  = math_fn.vector_add


# ---------------------------------------------------------------------------
# Activation: tanh
# ---------------------------------------------------------------------------

def tanh(z):
    """
    Hyperbolic tangent: output in (-1, 1), centered at zero.

    Clamps z to [-500, 500] to prevent overflow.

    Parameters
    ----------
    z : float — pre-activation value

    Returns
    -------
    float — tanh(z) in (-1, 1)
    """
    raise NotImplementedError("Your turn! Use math.tanh(z). Clamp first: z = max(-500, min(500, z))")


def tanh_derivative(a):
    """
    Derivative of tanh, given the tanh OUTPUT a (not the pre-activation z).

    If a = tanh(z), then  d(tanh)/dz = 1 - a².

    Parameters
    ----------
    a : float — the tanh activation output, tanh(z) for some z

    Returns
    -------
    float — derivative of tanh at this point
    """
    raise NotImplementedError("Your turn! Formula: 1.0 - a * a")


# ---------------------------------------------------------------------------
# SimpleMemoryCell
# ---------------------------------------------------------------------------

class SimpleMemoryCell:
    """
    The simplest recurrent unit: one tanh cell with hidden state.

    At each time step t:
        input_contrib  = W_input  @ x_t           (shape: hidden_size)
        hidden_contrib = W_hidden @ h_{t-1}        (shape: hidden_size)
        combined       = input_contrib + hidden_contrib + bias
        h_t            = tanh(combined)             (shape: hidden_size)

    Parameters
    ----------
    input_size  : int — dimensionality of each input vector
    hidden_size : int — dimensionality of the hidden state
    seed        : int — seed for reproducible Xavier initialisation
    """

    def __init__(self, input_size, hidden_size, seed=42):
        self.input_size  = input_size
        self.hidden_size = hidden_size

        rng = random.Random(seed)  # local RNG, doesn't pollute global state

        limit_input  = math.sqrt(6.0 / (input_size  + hidden_size))
        limit_hidden = math.sqrt(6.0 / (hidden_size + hidden_size))

        # W_input: [hidden_size x input_size]
        self.W_input = [
            [rng.uniform(-limit_input, limit_input) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        # W_hidden: [hidden_size x hidden_size]
        self.W_hidden = [
            [rng.uniform(-limit_hidden, limit_hidden) for _ in range(hidden_size)]
            for _ in range(hidden_size)
        ]
        self.bias = [0.0] * hidden_size
        self.hidden_state = [0.0] * hidden_size

    def step(self, input_vec):
        """
        Process one time step: update and return the hidden state.

        Computes:
            h_t = tanh(W_input @ x_t + W_hidden @ h_{t-1} + bias)

        Updates self.hidden_state in place. Returns a COPY.

        Parameters
        ----------
        input_vec : list of float, length = input_size

        Returns
        -------
        list of float — new hidden state h_t, length = hidden_size
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  1. input_contrib  = [dot_product(row, input_vec) for row in self.W_input]\n"
            "  2. hidden_contrib = [dot_product(row, self.hidden_state) for row in self.W_hidden]\n"
            "  3. combined = vector_add(vector_add(input_contrib, hidden_contrib), self.bias)\n"
            "  4. self.hidden_state = [tanh(z) for z in combined]\n"
            "  5. return list(self.hidden_state)  # return a copy, not the list itself"
        )

    def reset(self):
        """Zero the hidden state (call between independent sequences)."""
        self.hidden_state = [0.0] * self.hidden_size

    def process_sequence(self, sequence):
        """
        Process a full sequence from scratch.

        Resets hidden state first, then calls step() for each element.

        Parameters
        ----------
        sequence : list of list of float — one input vector per time step

        Returns
        -------
        list of list of float — one hidden state per time step
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Call self.reset(), then loop through sequence calling self.step().\n"
            "  Collect and return all hidden states."
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cell = SimpleMemoryCell(input_size=2, hidden_size=4, seed=42)

    # Test: same input, different history → different output
    out1 = cell.step([1.0, 0.0])
    cell.reset()
    cell.step([0.0, 1.0])
    cell.step([0.0, 1.0])
    out2 = cell.step([1.0, 0.0])

    print(f"Fresh cell, input=[1,0]:           {[round(v,4) for v in out1]}")
    print(f"After history [0,1],[0,1], same input: {[round(v,4) for v in out2]}")
    print(f"Outputs differ: {out1 != out2}  (should be True — that's the whole point!)")
