"""
Chapter 3 — Mamba / Selective State Space Model (SSM)
=======================================================

What this module teaches
-------------------------
State Space Models (SSMs) are a family of sequence models that maintain a
*hidden state* which is updated as each new token arrives. Unlike Transformers
(which attend over all past tokens simultaneously), an SSM processes the
sequence step-by-step like an RNN but with a structured, learnable state
transition matrix.

Mamba (Gu & Dao 2023) adds a key innovation over plain SSMs: **selectivity**.
A small input-dependent gate δ (delta) controls *how much* of the new input
to incorporate into the state at each step — the model learns when to update
its state and when to hold it fixed.

Core equations (this simplified version)
-----------------------------------------
  At each time step t with input x_t:

  1. delta_t = sigmoid(W_select @ x_t + b_select)
     — input-dependent gate ∈ (0, 1)^state_dim

  2. state_update = A @ state_{t-1} + B @ x_t
     — candidate new state (linear SSM update)

  3. state_t = delta_t ⊙ state_update + (1 - delta_t) ⊙ state_{t-1}
     — selective blend: how much new vs old state to keep

  4. h_t = C @ state_t
     — project state to hidden representation

  5. output_t = W_out @ relu(h_t) + b_out
     — project hidden to output logits

Key components
--------------
  A  [state_dim × state_dim]   — state transition (near-identity init)
  B  [state_dim × input_size]  — input projection into state space
  C  [hidden_size × state_dim] — state to hidden projection
  W_select [state_dim × input_size] + b_select — selectivity gate
  W_out [output_size × hidden_size] + b_out — output head

Why near-identity for A?
    At init, A ≈ I means the model mostly preserves state, allowing gradients
    to flow backward through many steps without vanishing immediately.

Imports from Level A
--------------------
  dot_product   (01_math_foundations)
  vector_add    (01_math_foundations)
  relu          (02_single_neuron)
  relu_derivative (02_single_neuron)
  sigmoid       (02_single_neuron)
  softmax       (04_loss_function)
  cross_entropy_loss (04_loss_function)
"""

import importlib
import sys
import os
import random
import math

# ---- Level A imports -------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product
vector_add = math_fn.vector_add

neuron_mod = importlib.import_module('02_single_neuron')
relu = neuron_mod.relu
relu_derivative = neuron_mod.relu_derivative
sigmoid = neuron_mod.sigmoid

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _mat_vec(matrix, vec):
    """matrix @ vec — row-wise dot products."""
    return [dot_product(row, vec) for row in matrix]


def _vec_scale(v, s):
    return [vi * s for vi in v]


def _vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_hadamard(a, b):
    """Element-wise (Hadamard) product a ⊙ b."""
    return [ai * bi for ai, bi in zip(a, b)]


def _zeros(n):
    return [0.0] * n


def _rand_matrix(rows, cols, rng, scale=0.1):
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def _rand_vec(n, rng, scale=0.01):
    return [rng.gauss(0, scale) for _ in range(n)]


def _identity_like(n, rng, noise=0.01):
    """Return an n×n near-identity matrix with small Gaussian noise."""
    return [
        [(1.0 if i == j else 0.0) + rng.gauss(0, noise)
         for j in range(n)]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# MambaModel
# ---------------------------------------------------------------------------

class MambaModel:
    """Simplified Selective State Space Model (Mamba-style).

    Parameters
    ----------
    input_size  : int — size of each input token vector (e.g. one-hot vocab)
    hidden_size : int — size of the hidden representation (output of C @ state)
    output_size : int — number of output classes
    state_dim   : int — dimensionality of the recurrent state vector (default 16)
    seed        : int — random seed for reproducibility (default 42)
    """

    def __init__(self, input_size, hidden_size, output_size, state_dim=16, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_dim = state_dim

        rng = random.Random(seed)

        # State-space matrices
        self.A = _identity_like(state_dim, rng, noise=0.01)           # [state_dim × state_dim]
        self.B = _rand_matrix(state_dim, input_size, rng, scale=0.1)  # [state_dim × input_size]
        self.C = _rand_matrix(hidden_size, state_dim, rng, scale=0.1) # [hidden_size × state_dim]

        # Selectivity gate
        self.W_select = _rand_matrix(state_dim, input_size, rng, scale=0.1)  # [state_dim × input_size]
        self.b_select = _zeros(state_dim)

        # Output head
        self.W_out = _rand_matrix(output_size, hidden_size, rng, scale=0.1)  # [output_size × hidden_size]
        self.b_out = _zeros(output_size)

        # Recurrent state — persists across forward_step calls
        self.state = _zeros(state_dim)

        # Cache for training
        self._last_h = None
        self._last_relu_h = None

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def forward_step(self, x):
        """Process one input token and update state.

        Parameters
        ----------
        x : list[float] — input vector of length input_size

        Returns
        -------
        list[float] — output logit vector of length output_size
        """
        # 1. Selectivity gate: delta ∈ (0,1)^state_dim
        gate_pre = _vec_add(_mat_vec(self.W_select, x), self.b_select)
        delta = [sigmoid(g) for g in gate_pre]

        # 2. Candidate state update: A @ state + B @ x
        state_candidate = _vec_add(_mat_vec(self.A, self.state), _mat_vec(self.B, x))

        # 3. Selective blend: new_state = delta ⊙ candidate + (1-delta) ⊙ old_state
        one_minus_delta = [1.0 - d for d in delta]
        new_state = _vec_add(
            _vec_hadamard(delta, state_candidate),
            _vec_hadamard(one_minus_delta, self.state)
        )
        self.state = new_state

        # 4. Project state to hidden: h = C @ state
        h = _mat_vec(self.C, self.state)
        self._last_h = h

        # 5. Output: W_out @ relu(h) + b_out
        relu_h = [relu(hi) for hi in h]
        self._last_relu_h = relu_h
        output = _vec_add(_mat_vec(self.W_out, relu_h), self.b_out)

        return output

    # ------------------------------------------------------------------
    # Sequence processing
    # ------------------------------------------------------------------

    def forward(self, sequence):
        """Reset state, process entire sequence, return all outputs.

        Parameters
        ----------
        sequence : list[list[float]] — list of input vectors

        Returns
        -------
        list[list[float]] — output vectors, one per time step
        """
        self.reset()
        outputs = []
        for x in sequence:
            out = self.forward_step(x)
            outputs.append(out)
        return outputs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, input_seq, targets, lr=0.01):
        """Train on one sequence using analytical output-layer gradients.

        Analytical gradients are applied to W_out and b_out.
        The state-space matrices (A, B, C, W_select) receive a small
        numerical gradient update via finite differences on C and W_select
        to propagate learning through the state path.

        Parameters
        ----------
        input_seq : list[list[float]] — input sequence
        targets   : list[int] — integer class indices, one per step
        lr        : float — learning rate

        Returns
        -------
        float — mean cross-entropy loss for the sequence
        """
        outputs = self.forward(input_seq)
        total_loss = 0.0
        n = len(input_seq)

        # ---- Analytical update for W_out and b_out ----------------------
        # Re-run forward caching relu_h per step
        relu_h_cache = []
        self.reset()
        for x in input_seq:
            self.forward_step(x)
            relu_h_cache.append(list(self._last_relu_h))

        for step_idx, (output, target_idx) in enumerate(zip(outputs, targets)):
            sm = softmax(output)
            one_hot = [1.0 if i == target_idx else 0.0 for i in range(self.output_size)]
            total_loss += cross_entropy_loss(sm, one_hot)

            # dL/d_output[j] = sm[j] - one_hot[j]  (softmax + cross-entropy)
            d_out = [sm[j] - one_hot[j] for j in range(self.output_size)]

            rh = relu_h_cache[step_idx]

            # dL/dW_out[i][j] = d_out[i] * relu_h[j]
            for i in range(self.output_size):
                for j in range(self.hidden_size):
                    self.W_out[i][j] -= lr * d_out[i] * rh[j]
            # dL/db_out[i] = d_out[i]
            for i in range(self.output_size):
                self.b_out[i] -= lr * d_out[i]

        # ---- Numerical gradient for C (hidden projection) ---------------
        c_lr = lr * 0.1
        eps = 1e-4

        def _seq_loss_c():
            outs = self.forward(input_seq)
            loss = 0.0
            for out, tgt in zip(outs, targets):
                sm = softmax(out)
                oh = [1.0 if i == tgt else 0.0 for i in range(self.output_size)]
                loss += cross_entropy_loss(sm, oh)
            return loss / n

        for i in range(self.hidden_size):
            for j in range(self.state_dim):
                orig = self.C[i][j]
                self.C[i][j] = orig + eps
                lp = _seq_loss_c()
                self.C[i][j] = orig - eps
                lm = _seq_loss_c()
                self.C[i][j] = orig
                self.C[i][j] -= c_lr * (lp - lm) / (2 * eps)

        # ---- Numerical gradient for W_select (selectivity gate) ---------
        ws_lr = lr * 0.05
        for i in range(self.state_dim):
            for j in range(self.input_size):
                orig = self.W_select[i][j]
                self.W_select[i][j] = orig + eps
                lp = _seq_loss_c()
                self.W_select[i][j] = orig - eps
                lm = _seq_loss_c()
                self.W_select[i][j] = orig
                self.W_select[i][j] -= ws_lr * (lp - lm) / (2 * eps)

        return total_loss / n

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, input_seq):
        """Return predicted class index per step.

        Parameters
        ----------
        input_seq : list[list[float]]

        Returns
        -------
        list[int] — argmax of softmax(output) for each step
        """
        outputs = self.forward(input_seq)
        preds = []
        for out in outputs:
            probs = softmax(out)
            preds.append(probs.index(max(probs)))
        return preds

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Zero the recurrent state."""
        self.state = _zeros(self.state_dim)

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def get_params_count(self):
        """Total number of scalar parameters."""
        # A: state_dim × state_dim
        p = self.state_dim * self.state_dim
        # B: state_dim × input_size
        p += self.state_dim * self.input_size
        # C: hidden_size × state_dim
        p += self.hidden_size * self.state_dim
        # W_select + b_select: state_dim × input_size + state_dim
        p += self.state_dim * self.input_size + self.state_dim
        # W_out + b_out: output_size × hidden_size + output_size
        p += self.output_size * self.hidden_size + self.output_size
        return p


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3 — Mamba / Selective SSM Demo")
    print("=" * 60)

    INPUT_SIZE = 8
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4
    STATE_DIM = 16

    model = MambaModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                       state_dim=STATE_DIM, seed=0)

    print(f"\nTotal params : {model.get_params_count()}")

    import random
    rng = random.Random(7)
    seq = [[1.0 if i == rng.randint(0, INPUT_SIZE - 1) else 0.0
            for i in range(INPUT_SIZE)] for _ in range(5)]
    targets = [rng.randint(0, OUTPUT_SIZE - 1) for _ in range(5)]

    print("\n--- Forward pass ---")
    outputs = model.forward(seq)
    for step, out in enumerate(outputs):
        probs = softmax(out)
        print(f"  Step {step}: max_logit={max(out):.4f}  pred={probs.index(max(probs))}")

    print("\n--- State after full sequence ---")
    print(f"  state[:4] = {[f'{s:.4f}' for s in model.state[:4]]}")

    print("\n--- Reset ---")
    model.reset()
    print(f"  state sum after reset = {sum(abs(s) for s in model.state):.6f}")

    print("\n--- Training (5 steps) ---")
    for epoch in range(5):
        loss = model.train_step(seq, targets, lr=0.05)
        print(f"  Epoch {epoch + 1}: loss={loss:.4f}")

    print("\n--- Predictions ---")
    preds = model.predict(seq)
    print(f"  Predicted : {preds}")
    print(f"  Targets   : {targets}")
    print("\nDone.")
