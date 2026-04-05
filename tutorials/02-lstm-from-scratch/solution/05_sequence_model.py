"""
Chapter 5 — Trainable LSTM Sequence Model with BPTT Through Gates
==================================================================

What this module teaches
-------------------------
This is the most mathematically complex chapter so far. We build a complete,
trainable LSTM model that processes variable-length sequences and learns via
Backpropagation Through Time (BPTT) — all from scratch, no NumPy.

The LSTM forward pass computes forget, input, output gates and a candidate
at each time step, updating cell state and hidden state. The backward pass
unrolls through time, propagating gradients through every gate.

The cell state "highway" for gradient flow
--------------------------------------------
The key reason LSTMs solve vanishing gradients is the cell state update:

    c_t = f_t * c_{t-1} + i_t * candidate_t

During backprop, the gradient flows through the forget gate:

    dc_{t-1} = dc_t * f_t

If the forget gate is near 1.0 (remember everything), the gradient passes
through nearly unchanged — no exponential decay, no vanishing. This is the
"gradient highway" that makes LSTMs trainable over hundreds of time steps.

BPTT through LSTM gates
-------------------------
Each gate has its own gradient path:

  - Forget gate:  d_f = dc * c_{t-1} * sigmoid'(f)
  - Input gate:   d_i = dc * candidate * sigmoid'(i)
  - Output gate:  d_o = dh * tanh(c) * sigmoid'(o)
  - Candidate:    d_c = dc * input_gate * tanh'(candidate)

All four paths contribute to dW and db updates, and all four feed back into
dh_next and dc_next for the previous time step.

Chapter roadmap
----------------
  Chapter 1: Recurrence — hidden state, SimpleMemoryCell
  Chapter 2: Vanilla RNN — forward through time and BPTT
  Chapter 3: Vanishing gradients — demonstration and analysis
  Chapter 4: LSTM cell — gated memory (forward only)
  Chapter 5 (this file): Trainable LSTM sequence model with full BPTT
"""

import importlib
import sys
import os
import math
import random

# ---------------------------------------------------------------------------
# Import shared primitives from Level A and Level B
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(__file__))

math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product
vector_add = math_fn.vector_add

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss

neuron_mod = importlib.import_module('02_single_neuron')
sigmoid = neuron_mod.sigmoid

recurrence_mod = importlib.import_module('01_recurrence')
tanh = recurrence_mod.tanh
tanh_derivative = recurrence_mod.tanh_derivative


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid_derivative(a):
    """Derivative of sigmoid given the sigmoid OUTPUT a.  d/dz = a * (1 - a)."""
    return a * (1 - a)


def outer_product(a, b):
    """Outer product of vectors a and b: result[i][j] = a[i] * b[j]."""
    return [[ai * bj for bj in b] for ai in a]


def elementwise_mul(a, b):
    """Hadamard (element-wise) product of two vectors."""
    return [ai * bi for ai, bi in zip(a, b)]


# ---------------------------------------------------------------------------
# LSTMSequenceModel — trainable LSTM with output layer and full BPTT
# ---------------------------------------------------------------------------

class LSTMSequenceModel:
    """
    A complete LSTM sequence model: LSTM layer + linear output layer.

    Processes variable-length sequences, produces an output (logits) at every
    time step, and trains via BPTT through all LSTM gates.

    Parameters
    ----------
    input_size  : int — dimensionality of input vectors at each time step
    hidden_size : int — number of LSTM hidden units
    output_size : int — number of output classes
    seed        : int — random seed for reproducible Xavier initialisation
    """

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        combined_size = input_size + hidden_size

        rng = random.Random(seed)

        # Xavier (Glorot) uniform initialisation
        limit_lstm = math.sqrt(6.0 / (combined_size + hidden_size))
        limit_out = math.sqrt(6.0 / (hidden_size + output_size))

        def make_matrix(rows, cols, limit):
            return [[rng.uniform(-limit, limit) for _ in range(cols)]
                    for _ in range(rows)]

        # LSTM gate weights: each [hidden_size x combined_size]
        self.W_f = make_matrix(hidden_size, combined_size, limit_lstm)
        self.W_i = make_matrix(hidden_size, combined_size, limit_lstm)
        self.W_o = make_matrix(hidden_size, combined_size, limit_lstm)
        self.W_c = make_matrix(hidden_size, combined_size, limit_lstm)

        # LSTM gate biases: each [hidden_size]
        self.b_f = [1.0] * hidden_size   # forget bias = 1.0 (remember by default)
        self.b_i = [0.0] * hidden_size
        self.b_o = [0.0] * hidden_size
        self.b_c = [0.0] * hidden_size

        # Output layer: W_out [output_size x hidden_size], b_out [output_size]
        self.W_out = make_matrix(output_size, hidden_size, limit_out)
        self.b_out = [0.0] * output_size

        # Cache for backward pass (filled during forward)
        self._cache = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, sequence):
        """
        Run the LSTM over a sequence, producing output logits at each step.

        Parameters
        ----------
        sequence : list of list of float
            Each element is one time step's input vector (length = input_size).

        Returns
        -------
        list of list of float
            Output logits at each time step (length = output_size each).
        """
        h = [0.0] * self.hidden_size
        c = [0.0] * self.hidden_size
        self._cache = []
        outputs = []

        for x_t in sequence:
            # 1. Concatenate input with previous hidden state
            combined = list(x_t) + list(h)

            # 2. Compute gates
            forget_gate = [sigmoid(dot_product(self.W_f[j], combined) + self.b_f[j])
                           for j in range(self.hidden_size)]
            input_gate = [sigmoid(dot_product(self.W_i[j], combined) + self.b_i[j])
                          for j in range(self.hidden_size)]
            output_gate = [sigmoid(dot_product(self.W_o[j], combined) + self.b_o[j])
                           for j in range(self.hidden_size)]
            candidate = [tanh(dot_product(self.W_c[j], combined) + self.b_c[j])
                         for j in range(self.hidden_size)]

            # 3. Update cell state: c_t = f_t * c_{t-1} + i_t * candidate
            c_prev = list(c)
            c = vector_add(elementwise_mul(forget_gate, c),
                           elementwise_mul(input_gate, candidate))

            # 4. Update hidden state: h_t = o_t * tanh(c_t)
            tanh_cell = [tanh(ci) for ci in c]
            h = elementwise_mul(output_gate, tanh_cell)

            # 5. Compute output logits: W_out @ h + b_out
            output_logits = [dot_product(self.W_out[k], h) + self.b_out[k]
                             for k in range(self.output_size)]

            # 6. Cache everything needed for backward pass
            self._cache.append({
                'combined': combined,
                'forget_gate': forget_gate,
                'input_gate': input_gate,
                'output_gate': output_gate,
                'candidate': candidate,
                'cell_state': list(c),
                'cell_prev': c_prev,
                'hidden_state': list(h),
                'tanh_cell': tanh_cell,
                'output_logits': output_logits,
            })

            outputs.append(output_logits)

        return outputs

    # ------------------------------------------------------------------
    # Predict (argmax of softmax)
    # ------------------------------------------------------------------

    def predict(self, sequence):
        """
        Run forward pass, apply softmax, return argmax index at each step.

        Parameters
        ----------
        sequence : list of list of float

        Returns
        -------
        list of int — predicted class index at each time step
        """
        outputs = self.forward(sequence)
        predictions = []
        for logits in outputs:
            probs = softmax(logits)
            best = max(range(len(probs)), key=lambda i: probs[i])
            predictions.append(best)
        return predictions

    # ------------------------------------------------------------------
    # Backward pass — BPTT through LSTM gates
    # ------------------------------------------------------------------

    def backward(self, targets):
        """
        Backpropagation Through Time through all LSTM gates.

        Must be called after forward(). Computes gradients for all weights
        and biases, and returns the total cross-entropy loss.

        Parameters
        ----------
        targets : list of int
            Target class index at each time step.

        Returns
        -------
        float — total cross-entropy loss summed over all time steps
        """
        T = len(self._cache)
        hs = self.hidden_size
        cs = self.input_size + self.hidden_size  # combined_size

        # Initialise gradient accumulators to zero
        dW_f = [[0.0] * cs for _ in range(hs)]
        dW_i = [[0.0] * cs for _ in range(hs)]
        dW_o = [[0.0] * cs for _ in range(hs)]
        dW_c = [[0.0] * cs for _ in range(hs)]
        db_f = [0.0] * hs
        db_i = [0.0] * hs
        db_o = [0.0] * hs
        db_c = [0.0] * hs

        dW_out = [[0.0] * hs for _ in range(self.output_size)]
        db_out = [0.0] * self.output_size

        # Gradients flowing backward from future time step
        dh_next = [0.0] * hs
        dc_next = [0.0] * hs

        total_loss = 0.0

        # Walk backward through time
        for t in reversed(range(T)):
            cache = self._cache[t]
            logits = cache['output_logits']
            combined = cache['combined']
            forget_gate = cache['forget_gate']
            input_gate = cache['input_gate']
            output_gate = cache['output_gate']
            candidate = cache['candidate']
            cell_state = cache['cell_state']
            cell_prev = cache['cell_prev']
            hidden_state = cache['hidden_state']
            tanh_cell = cache['tanh_cell']

            # --- Step 1: Output gradient ---
            probs = softmax(logits)
            one_hot = [0.0] * self.output_size
            one_hot[targets[t]] = 1.0
            total_loss += cross_entropy_loss(probs, one_hot)

            # dL/d(logits) = probs - one_hot (softmax + cross-entropy shortcut)
            dout = [probs[k] - one_hot[k] for k in range(self.output_size)]

            # --- Step 2: Accumulate output layer gradients ---
            for k in range(self.output_size):
                db_out[k] += dout[k]
                for j in range(hs):
                    dW_out[k][j] += dout[k] * hidden_state[j]

            # --- Step 3: Gradient into hidden state ---
            # dh = W_out^T @ dout + dh_next
            dh = [0.0] * hs
            for j in range(hs):
                for k in range(self.output_size):
                    dh[j] += self.W_out[k][j] * dout[k]
                dh[j] += dh_next[j]

            # --- Step 4: Gradient into cell state ---
            # dc = dh * output_gate * tanh'(cell) + dc_next
            dc = [0.0] * hs
            for j in range(hs):
                dc[j] = dh[j] * output_gate[j] * tanh_derivative(tanh_cell[j]) + dc_next[j]

            # --- Step 5: Gate gradients ---
            d_output_gate = [dh[j] * tanh_cell[j] * sigmoid_derivative(output_gate[j])
                             for j in range(hs)]
            d_forget_gate = [dc[j] * cell_prev[j] * sigmoid_derivative(forget_gate[j])
                             for j in range(hs)]
            d_input_gate = [dc[j] * candidate[j] * sigmoid_derivative(input_gate[j])
                            for j in range(hs)]
            d_candidate = [dc[j] * input_gate[j] * tanh_derivative(candidate[j])
                           for j in range(hs)]

            # --- Step 6: Accumulate gate weight gradients ---
            for j in range(hs):
                for c_idx in range(cs):
                    dW_f[j][c_idx] += d_forget_gate[j] * combined[c_idx]
                    dW_i[j][c_idx] += d_input_gate[j] * combined[c_idx]
                    dW_o[j][c_idx] += d_output_gate[j] * combined[c_idx]
                    dW_c[j][c_idx] += d_candidate[j] * combined[c_idx]
                db_f[j] += d_forget_gate[j]
                db_i[j] += d_input_gate[j]
                db_o[j] += d_output_gate[j]
                db_c[j] += d_candidate[j]

            # --- Step 7: Compute dh_next from all gate weight contributions ---
            # The hidden portion of combined is indices [input_size:]
            dh_next = [0.0] * hs
            for j in range(hs):
                for h_idx in range(hs):
                    idx = self.input_size + h_idx
                    dh_next[h_idx] += (self.W_f[j][idx] * d_forget_gate[j]
                                       + self.W_i[j][idx] * d_input_gate[j]
                                       + self.W_o[j][idx] * d_output_gate[j]
                                       + self.W_c[j][idx] * d_candidate[j])

            # --- Step 8: Cell gradient flows through forget gate ---
            dc_next = [dc[j] * forget_gate[j] for j in range(hs)]

        # Store gradients for SGD update
        self._grads = {
            'dW_f': dW_f, 'dW_i': dW_i, 'dW_o': dW_o, 'dW_c': dW_c,
            'db_f': db_f, 'db_i': db_i, 'db_o': db_o, 'db_c': db_c,
            'dW_out': dW_out, 'db_out': db_out,
        }

        return total_loss

    # ------------------------------------------------------------------
    # SGD update
    # ------------------------------------------------------------------

    def _sgd_update(self, lr):
        """Apply SGD: param -= lr * grad for all parameters."""
        g = self._grads

        # LSTM gate weights and biases
        for gate_W, gate_dW in [(self.W_f, g['dW_f']), (self.W_i, g['dW_i']),
                                 (self.W_o, g['dW_o']), (self.W_c, g['dW_c'])]:
            for j in range(len(gate_W)):
                for k in range(len(gate_W[0])):
                    gate_W[j][k] -= lr * gate_dW[j][k]

        for gate_b, gate_db in [(self.b_f, g['db_f']), (self.b_i, g['db_i']),
                                 (self.b_o, g['db_o']), (self.b_c, g['db_c'])]:
            for j in range(len(gate_b)):
                gate_b[j] -= lr * gate_db[j]

        # Output layer
        for k in range(self.output_size):
            for j in range(self.hidden_size):
                self.W_out[k][j] -= lr * g['dW_out'][k][j]
            self.b_out[k] -= lr * g['db_out'][k]

    # ------------------------------------------------------------------
    # Train step and train loop
    # ------------------------------------------------------------------

    def train_step(self, sequence, targets, lr=0.05):
        """
        One training step: forward, backward, SGD update.

        Parameters
        ----------
        sequence : list of list of float — input sequence
        targets  : list of int — target class at each time step
        lr       : float — learning rate

        Returns
        -------
        float — total loss for this sequence
        """
        self.forward(sequence)
        loss = self.backward(targets)
        self._sgd_update(lr)
        return loss

    def train(self, dataset, epochs, lr=0.05, verbose=True):
        """
        Train on a dataset for multiple epochs.

        Parameters
        ----------
        dataset : list of (sequence, targets) tuples
        epochs  : int — number of full passes over the dataset
        lr      : float — learning rate
        verbose : bool — print epoch losses

        Returns
        -------
        list of float — average loss per epoch
        """
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for sequence, targets in dataset:
                total_loss += self.train_step(sequence, targets, lr)
            avg_loss = total_loss / len(dataset)
            epoch_losses.append(avg_loss)
            if verbose:
                print(f"  Epoch {epoch + 1:3d}/{epochs}  loss = {avg_loss:.6f}")
        return epoch_losses


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("Chapter 5 — Trainable LSTM Sequence Model")
    print("=" * 65)
    print()

    # Simple task: given a one-hot sequence [A, B, C], predict the identity
    # of each element (class 0, 1, 2).
    input_size = 3
    hidden_size = 10
    output_size = 3

    model = LSTMSequenceModel(input_size, hidden_size, output_size, seed=42)

    # Training data: three short sequences
    dataset = [
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2]),  # A B C -> 0 1 2
        ([[0, 1, 0], [0, 0, 1], [1, 0, 0]], [1, 2, 0]),  # B C A -> 1 2 0
        ([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [2, 0, 1]),  # C A B -> 2 0 1
    ]

    print("Training on 3 sequences for 100 epochs...")
    print("-" * 45)
    losses = model.train(dataset, epochs=100, lr=0.05, verbose=False)

    # Show loss at key points
    print(f"  Epoch   1  loss = {losses[0]:.6f}")
    print(f"  Epoch  25  loss = {losses[24]:.6f}")
    print(f"  Epoch  50  loss = {losses[49]:.6f}")
    print(f"  Epoch 100  loss = {losses[99]:.6f}")
    print()

    # Test predictions
    print("Predictions after training:")
    print("-" * 45)
    for seq, tgt in dataset:
        preds = model.predict(seq)
        labels = ['A', 'B', 'C']
        seq_str = ' '.join(labels[i] for i, v in enumerate(seq) for j, x in enumerate(v) if x == 1)
        pred_str = ' '.join(str(p) for p in preds)
        tgt_str = ' '.join(str(t) for t in tgt)
        correct = "OK" if preds == tgt else "MISS"
        print(f"  Input: [{seq_str}]  Target: [{tgt_str}]  Pred: [{pred_str}]  {correct}")

    print()
    print("Key insight: the cell state highway lets gradients flow")
    print("through forget gates without vanishing — dc_next = dc * f_t.")
    print("This is why LSTMs can learn long-range dependencies.")
