"""
Chapter 2 — Vanilla RNN with Backpropagation Through Time (BPTT)
================================================================

What this module teaches
-------------------------
This chapter builds a full vanilla RNN that can be *trained* end-to-end.
Chapter 1 gave us a recurrent cell that carries hidden state forward through
a sequence. But forward propagation alone is useless without learning. To
learn, we need gradients — and in a recurrent network, gradients must flow
*backward through time*.

Backpropagation Through Time (BPTT)
-------------------------------------
BPTT is the standard algorithm for training RNNs. It "unrolls" the recurrent
network across all time steps, treating it as a very deep feedforward network
where every layer shares the same weights. Then ordinary backpropagation is
applied to this unrolled graph.

The critical insight: at each time step t, the hidden state gradient dh has
TWO sources:
  1. The gradient from the output loss at time t  (dh_from_output)
  2. The gradient propagated backward from time t+1  (dh_next)

These are summed: dh = dh_from_output + dh_next. This accumulation is what
makes BPTT work — it is the mechanism by which a loss at the end of a
sequence can influence weights that were used at the beginning.

The vanishing gradient problem
-------------------------------
Because dh_next passes through a tanh derivative (which is <= 1) and a
matrix multiplication (W_hidden) at every step, gradients can shrink
exponentially as they travel backward. After ~10-20 steps the gradient
reaching early positions is negligible. This is the vanishing gradient
problem, and it is the reason LSTMs and GRUs were invented (Chapter 3).

Implementation details
-----------------------
  - RNNCell: one recurrent cell with forward() that caches activations
  - RNN: full network with output projection, forward through a sequence,
    BPTT backward pass, and SGD training step
  - All pure Python, no NumPy — lists of floats for vectors, lists of lists
    for matrices

Chapter roadmap
---------------
  Chapter 1: Recurrence — hidden state, SimpleMemoryCell
  Chapter 2 (this file): Vanilla RNN — forward through time and BPTT
  Chapter 3: LSTM — gated memory, forget/input/output gates
  Chapter 4: Attention — weighting past states by relevance
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

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss

recurrence_mod = importlib.import_module('01_recurrence')
tanh = recurrence_mod.tanh
tanh_derivative = recurrence_mod.tanh_derivative


# ---------------------------------------------------------------------------
# Helper: outer product of two vectors → matrix
# ---------------------------------------------------------------------------

def outer_product(a, b):
    """
    Compute the outer product of vectors a and b.

    result[i][j] = a[i] * b[j]

    Returns a matrix of shape [len(a) x len(b)].
    """
    return [[ai * bj for bj in b] for ai in a]


# ---------------------------------------------------------------------------
# RNNCell — a single recurrent cell with gradient storage
# ---------------------------------------------------------------------------

class RNNCell:
    """
    A single vanilla RNN cell that caches activations for BPTT.

    Forward computation:
        h_t = tanh(W_input @ x_t + W_hidden @ h_{t-1} + bias)

    Unlike SimpleMemoryCell from Chapter 1, this cell:
      - Stores input_vec, prev_hidden, and new_h after each forward() call
        so that the backward pass can compute gradients
      - Maintains gradient accumulators (dW_input, dW_hidden, dbias) that
        are summed across all time steps before a single weight update

    Parameters
    ----------
    input_size  : int — dimensionality of input vectors
    hidden_size : int — dimensionality of the hidden state
    seed        : int — seed for reproducible Xavier initialisation
    """

    def __init__(self, input_size, hidden_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size

        rng = random.Random(seed)

        # Xavier (Glorot) uniform initialisation
        limit_input = math.sqrt(6.0 / (input_size + hidden_size))
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

        # Bias: [hidden_size], initialised to zero
        self.bias = [0.0] * hidden_size

        # Gradient accumulators (same shapes as weights)
        self.dW_input = [[0.0] * input_size for _ in range(hidden_size)]
        self.dW_hidden = [[0.0] * hidden_size for _ in range(hidden_size)]
        self.dbias = [0.0] * hidden_size

        # Cached activations (set by forward, read by backward)
        self.input_vec = None
        self.prev_hidden = None
        self.new_h = None

    def forward(self, input_vec, prev_hidden):
        """
        One forward step of the RNN cell.

        Computes h_t = tanh(W_input @ x_t + W_hidden @ h_{t-1} + bias)
        and caches (input_vec, prev_hidden, new_h) for BPTT.

        Parameters
        ----------
        input_vec   : list[float] — current input, length = input_size
        prev_hidden : list[float] — previous hidden state, length = hidden_size

        Returns
        -------
        list[float] — new hidden state h_t, length = hidden_size
        """
        # Cache for backward pass
        self.input_vec = list(input_vec)
        self.prev_hidden = list(prev_hidden)

        # Linear projections
        input_contrib = [dot_product(row, input_vec) for row in self.W_input]
        hidden_contrib = [dot_product(row, prev_hidden) for row in self.W_hidden]

        # Combine and apply tanh
        combined = vector_add(vector_add(input_contrib, hidden_contrib), self.bias)
        self.new_h = [tanh(z) for z in combined]

        return list(self.new_h)

    def zero_gradients(self):
        """Reset all gradient accumulators to zero."""
        self.dW_input = [[0.0] * self.input_size for _ in range(self.hidden_size)]
        self.dW_hidden = [[0.0] * self.hidden_size for _ in range(self.hidden_size)]
        self.dbias = [0.0] * self.hidden_size


# ---------------------------------------------------------------------------
# RNN — full network with output layer, forward, BPTT, and training
# ---------------------------------------------------------------------------

class RNN:
    """
    A complete vanilla RNN for sequence classification / prediction.

    Architecture (per time step):
        h_t = RNNCell.forward(x_t, h_{t-1})
        output_t = W_output @ h_t + b_output

    The output logits are converted to probabilities via softmax during the
    backward pass (to compute the cross-entropy gradient).

    Parameters
    ----------
    input_size  : int — dimensionality of each input vector
    hidden_size : int — dimensionality of the hidden state
    output_size : int — number of output classes
    seed        : int — reproducible initialisation
    """

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Recurrent cell
        self.cell = RNNCell(input_size, hidden_size, seed=seed)

        # Output projection: [output_size x hidden_size]
        rng = random.Random(seed + 1)  # different seed from cell
        limit_output = math.sqrt(6.0 / (hidden_size + output_size))

        self.W_output = [
            [rng.uniform(-limit_output, limit_output) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]
        self.b_output = [0.0] * output_size

        # Output gradient accumulators
        self.dW_output = [[0.0] * hidden_size for _ in range(output_size)]
        self.db_output = [0.0] * output_size

        # Cached sequences (populated by forward, consumed by backward)
        self.inputs = []
        self.hiddens = []   # hiddens[0] = h_0 (initial zeros), hiddens[t+1] = h after step t
        self.outputs = []

    def forward(self, sequence):
        """
        Run the RNN forward through an entire sequence.

        Parameters
        ----------
        sequence : list of list[float]
            Each element is one time step's input vector (length = input_size).

        Returns
        -------
        (outputs_list, hidden_states_list) : tuple
            outputs_list    — list of raw logit vectors (one per time step)
            hidden_states_list — list of hidden states (one per time step)
        """
        T = len(sequence)

        # Initial hidden state: zeros
        h = [0.0] * self.hidden_size

        self.inputs = []
        self.hiddens = [list(h)]  # h_0 at index 0
        self.outputs = []

        outputs_list = []
        hidden_states_list = []

        for t in range(T):
            x_t = sequence[t]
            self.inputs.append(list(x_t))

            # Recurrent step
            h = self.cell.forward(x_t, h)
            self.hiddens.append(list(h))  # h_{t+1} at index t+1
            hidden_states_list.append(list(h))

            # Output projection
            output = [
                dot_product(self.W_output[i], h) + self.b_output[i]
                for i in range(self.output_size)
            ]
            self.outputs.append(output)
            outputs_list.append(list(output))

        return outputs_list, hidden_states_list

    def backward(self, targets):
        """
        Backpropagation Through Time (BPTT).

        Computes gradients of the cross-entropy loss with respect to all
        weights, accumulating across all time steps.

        The algorithm:
          For t = T-1 down to 0:
            1. Output gradient: probs = softmax(outputs[t]),
               dout = probs - one_hot(targets[t])
            2. Accumulate output weight gradients
            3. Compute dh from output layer
            4. Add dh from future time step (dh_next)
            5. Multiply by tanh derivative to get dh_raw
            6. Accumulate cell weight gradients
            7. Propagate dh_next backward for the next iteration

        Parameters
        ----------
        targets : list[int]
            Target class indices, one per time step. Length must match the
            number of time steps from the last forward() call.

        Returns
        -------
        float — total cross-entropy loss summed across all time steps
        """
        T = len(targets)
        total_loss = 0.0

        # dh_next: gradient flowing backward from future time steps
        dh_next = [0.0] * self.hidden_size

        for t in range(T - 1, -1, -1):
            # --- Output gradient ---
            probs = softmax(self.outputs[t])

            # One-hot target
            one_hot = [0.0] * self.output_size
            one_hot[targets[t]] = 1.0

            # Cross-entropy loss at this step
            total_loss += cross_entropy_loss(probs, one_hot)

            # Gradient of softmax + cross-entropy: dout = probs - one_hot
            dout = [probs[i] - one_hot[i] for i in range(self.output_size)]

            # --- Accumulate output weight gradients ---
            # dW_output += outer(dout, h_t)
            h_t = self.hiddens[t + 1]  # hidden state at time t (offset by 1)
            for i in range(self.output_size):
                for j in range(self.hidden_size):
                    self.dW_output[i][j] += dout[i] * h_t[j]
            # db_output += dout
            for i in range(self.output_size):
                self.db_output[i] += dout[i]

            # --- Hidden state gradient from output layer ---
            # dh_from_output = W_output^T @ dout
            dh_from_output = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                for i in range(self.output_size):
                    dh_from_output[j] += self.W_output[i][j] * dout[i]

            # --- Combine with gradient from future ---
            dh = [dh_from_output[j] + dh_next[j] for j in range(self.hidden_size)]

            # --- Through tanh: dh_raw = dh * tanh'(h_t) ---
            dh_raw = [dh[j] * tanh_derivative(h_t[j]) for j in range(self.hidden_size)]

            # --- Accumulate cell gradients ---
            x_t = self.inputs[t]
            h_prev = self.hiddens[t]  # h_{t-1}

            # dW_input += outer(dh_raw, x_t)
            for i in range(self.hidden_size):
                for j in range(self.cell.input_size):
                    self.cell.dW_input[i][j] += dh_raw[i] * x_t[j]

            # dW_hidden += outer(dh_raw, h_{t-1})
            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    self.cell.dW_hidden[i][j] += dh_raw[i] * h_prev[j]

            # dbias += dh_raw
            for i in range(self.hidden_size):
                self.cell.dbias[i] += dh_raw[i]

            # --- Propagate dh_next backward: W_hidden^T @ dh_raw ---
            dh_next = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                for i in range(self.hidden_size):
                    dh_next[j] += self.cell.W_hidden[i][j] * dh_raw[i]

        return total_loss

    def _zero_all_gradients(self):
        """Zero gradients for both the cell and the output layer."""
        self.cell.zero_gradients()
        self.dW_output = [[0.0] * self.hidden_size for _ in range(self.output_size)]
        self.db_output = [0.0] * self.output_size

    def train_step(self, sequence, targets, lr):
        """
        One complete training step: zero grads, forward, backward, SGD update.

        Parameters
        ----------
        sequence : list of list[float] — input sequence
        targets  : list[int] — target class indices (one per time step)
        lr       : float — learning rate

        Returns
        -------
        float — total cross-entropy loss for this sequence
        """
        # Zero all gradients
        self._zero_all_gradients()

        # Forward pass
        self.forward(sequence)

        # Backward pass (BPTT)
        loss = self.backward(targets)

        # SGD update: W -= lr * dW
        # Cell weights
        for i in range(self.hidden_size):
            for j in range(self.cell.input_size):
                self.cell.W_input[i][j] -= lr * self.cell.dW_input[i][j]
            for j in range(self.hidden_size):
                self.cell.W_hidden[i][j] -= lr * self.cell.dW_hidden[i][j]
            self.cell.bias[i] -= lr * self.cell.dbias[i]

        # Output weights
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.W_output[i][j] -= lr * self.dW_output[i][j]
            self.b_output[i] -= lr * self.db_output[i]

        return loss


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 02-lstm-from-scratch/solution/02_vanilla_rnn.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 2 — Vanilla RNN with BPTT")
    print("Training a sequence-to-sequence mapping from scratch")
    print("=" * 60)

    # Task: learn a simple sequence pattern
    # Input: 3-step sequence of 2D vectors
    # Target: at each step, predict one of 3 classes
    #
    # Pattern: [1,0] → class 0, [0,1] → class 1, [1,1] → class 2
    # The RNN should learn this mapping (and the hidden state helps
    # disambiguate when patterns overlap).

    input_size = 2
    hidden_size = 8
    output_size = 3

    rnn = RNN(input_size, hidden_size, output_size, seed=42)

    # Training data: a single sequence repeated
    sequence = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    targets = [0, 1, 2]

    print(f"\nInput sequence:  {sequence}")
    print(f"Target classes:  {targets}")
    print(f"Architecture:    input={input_size}, hidden={hidden_size}, output={output_size}")
    print(f"\nTraining for 50 steps with lr=0.1:")
    print("-" * 40)

    for step in range(50):
        loss = rnn.train_step(sequence, targets, lr=0.1)
        if step % 5 == 0 or step == 49:
            # Show predictions
            outputs, _ = rnn.forward(sequence)
            preds = [softmax(o) for o in outputs]
            pred_classes = [max(range(output_size), key=lambda i: p[i]) for p in preds]
            print(f"  step {step:3d}  loss={loss:.4f}  preds={pred_classes}  target={targets}")

    # Final predictions
    outputs, hiddens = rnn.forward(sequence)
    print("\nFinal output probabilities:")
    for t, out in enumerate(outputs):
        probs = softmax(out)
        print(f"  t={t}: {[round(p, 4) for p in probs]}  → class {max(range(output_size), key=lambda i: probs[i])}")

    print(f"\nHidden state dimensionality: {len(hiddens[0])}")
    print(f"Number of time steps processed: {len(hiddens)}")

    print("\n" + "=" * 60)
    print("Chapter 2 complete. Gradients flow backward through time.")
    print("Next: Chapter 3 — LSTM: gated memory to fight vanishing gradients.")
    print("=" * 60)
