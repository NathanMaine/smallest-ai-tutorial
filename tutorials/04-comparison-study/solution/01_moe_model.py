"""
Chapter 2 — Mixture of Experts (MoE)
======================================

What this module teaches
-------------------------
Mixture of Experts is an architecture that keeps a model large in *capacity*
(many expert sub-networks) while keeping each *inference* cheap: only a
small subset of experts (top_k) is activated for any given input.

Key ideas
---------
  Router     — A learned linear layer + softmax that assigns a probability
               to each expert. We select the top_k highest-probability
               experts and weight their outputs by those probabilities.

  Expert     — A small 2-layer MLP: input → hidden → output.
               Each expert specialises on a different part of the input space.

  Sparsity   — Only top_k experts fire per token. This means:
                 active_params << total_params
               The network is large (many experts can exist) but fast
               (few experts compute on any one input).

  Weighted sum — The final output is:
                   out = sum( gate_weight[k] * expert_k(x)  for k in top_k )

Architecture walkthrough
------------------------
  1. Input x (input_size) →  Router linear  → logits (num_experts)
  2. softmax(logits) → gate probabilities
  3. argsort descending → pick top_k expert indices
  4. Run each chosen expert on x → expert outputs (output_size each)
  5. Weighted sum over expert outputs using their gate probabilities (re-normalised)
  6. Return combined output

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


def _zeros(n):
    return [0.0] * n


def _rand_matrix(rows, cols, rng, scale=0.1):
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def _rand_vec(n, rng, scale=0.0):
    return [rng.gauss(0, scale) for _ in range(n)]


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class Expert:
    """A 2-layer MLP: input → hidden (ReLU) → output (linear).

    Parameters
    ----------
    input_size  : int — dimensionality of the input vector
    hidden_size : int — number of hidden units
    output_size : int — dimensionality of the output vector
    rng         : random.Random — seeded random instance for reproducibility
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, rng):
        # Layer 1 weights and biases: hidden_size x input_size
        self.W1 = _rand_matrix(hidden_size, input_size, rng, scale=0.1)
        self.b1 = _zeros(hidden_size)
        # Layer 2 weights and biases: output_size x hidden_size
        self.W2 = _rand_matrix(output_size, hidden_size, rng, scale=0.1)
        self.b2 = _zeros(output_size)

        # Cached intermediate values (set during forward pass)
        self.last_input = None
        self.last_z1 = None   # pre-activation hidden
        self.last_h = None    # post-activation hidden
        self.last_output = None

    def forward(self, x):
        """Run one forward pass.

        Parameters
        ----------
        x : list[float] — input vector of length input_size

        Returns
        -------
        list[float] — output vector of length output_size
        """
        self.last_input = x

        # Hidden layer: z1 = W1 @ x + b1, h = relu(z1)
        z1 = _vec_add(_mat_vec(self.W1, x), self.b1)
        self.last_z1 = z1
        h = [relu(zi) for zi in z1]
        self.last_h = h

        # Output layer: output = W2 @ h + b2
        output = _vec_add(_mat_vec(self.W2, h), self.b2)
        self.last_output = output
        return output

    @property
    def param_count(self):
        """Total number of scalar parameters in this expert."""
        h = len(self.W1)
        i = len(self.W1[0])
        o = len(self.W2)
        return h * i + h + o * h + o


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    """Linear layer + softmax gate probabilities.

    Maps an input vector to a probability distribution over num_experts.

    Parameters
    ----------
    input_size  : int
    num_experts : int
    rng         : random.Random
    """

    def __init__(self, input_size: int, num_experts: int, rng):
        self.W = _rand_matrix(num_experts, input_size, rng, scale=0.1)
        self.b = _zeros(num_experts)
        self.last_input = None
        self.last_logits = None
        self.last_probs = None

    def forward(self, x):
        """Compute gate probabilities.

        Parameters
        ----------
        x : list[float]

        Returns
        -------
        list[float] — probability for each expert (sums to 1)
        """
        self.last_input = x
        logits = _vec_add(_mat_vec(self.W, x), self.b)
        self.last_logits = logits
        probs = softmax(logits)
        self.last_probs = probs
        return probs

    def select_top_k(self, probs, top_k: int):
        """Return (indices, weights) for the top_k experts.

        Weights are re-normalised to sum to 1 among the selected experts.

        Parameters
        ----------
        probs : list[float] — softmax probabilities (length = num_experts)
        top_k : int

        Returns
        -------
        tuple(list[int], list[float]) — sorted indices and their re-normalised weights
        """
        indexed = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)
        top = indexed[:top_k]
        indices = [i for i, _ in top]
        weights_raw = [w for _, w in top]
        total = sum(weights_raw)
        weights = [w / total for w in weights_raw]
        return indices, weights

    @property
    def param_count(self):
        return len(self.W) * len(self.W[0]) + len(self.b)


# ---------------------------------------------------------------------------
# MoEModel
# ---------------------------------------------------------------------------

class MoEModel:
    """Mixture of Experts sequence model.

    Parameters
    ----------
    input_size  : int — size of each input token vector (one-hot)
    hidden_size : int — hidden units per expert
    output_size : int — number of output classes
    num_experts : int — total number of experts (default 4)
    top_k       : int — how many experts to activate per token (default 2)
    seed        : int — random seed for reproducibility (default 42)
    """

    def __init__(self, input_size, hidden_size, output_size,
                 num_experts=4, top_k=2, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.top_k = top_k

        rng = random.Random(seed)
        self.router = Router(input_size, num_experts, rng)
        self.experts = [
            Expert(input_size, hidden_size, output_size, rng)
            for _ in range(num_experts)
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """Forward pass for a single input vector.

        Parameters
        ----------
        x : list[float] — input token vector (length input_size)

        Returns
        -------
        list[float] — combined output vector (length output_size)
        tuple(list[int], list[float]) — selected expert indices and their weights
        """
        probs = self.router.forward(x)
        indices, weights = self.router.select_top_k(probs, self.top_k)

        # Weighted sum of selected experts' outputs
        combined = _zeros(self.output_size)
        for idx, w in zip(indices, weights):
            expert_out = self.experts[idx].forward(x)
            weighted = _vec_scale(expert_out, w)
            combined = _vec_add(combined, weighted)

        return combined, (indices, weights)

    def forward_sequence(self, sequence):
        """Apply forward to each step in a sequence.

        Parameters
        ----------
        sequence : list[list[float]] — list of input vectors

        Returns
        -------
        list of (output, routing_info) — one per time step
        """
        return [self.forward(x) for x in sequence]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, input_seq, targets, lr=0.01):
        """Single training step over a sequence.

        Uses analytical gradients for each expert's output layer (W2, b2)
        and numerical gradients for the router weights.

        Parameters
        ----------
        input_seq : list[list[float]] — input sequence
        targets   : list[int] — integer class indices, one per step
        lr        : float — learning rate

        Returns
        -------
        float — mean cross-entropy loss for the sequence
        """
        total_loss = 0.0
        n_steps = len(input_seq)

        # ---- analytical expert output-layer update ---------------------
        for x, target_idx in zip(input_seq, targets):
            probs = self.router.forward(x)
            indices, weights = self.router.select_top_k(probs, self.top_k)

            combined = _zeros(self.output_size)
            for idx, w in zip(indices, weights):
                expert_out = self.experts[idx].forward(x)
                combined = _vec_add(combined, _vec_scale(expert_out, w))

            sm = softmax(combined)
            one_hot = [1.0 if i == target_idx else 0.0 for i in range(self.output_size)]
            total_loss += cross_entropy_loss(sm, one_hot)

            # Gradient of cross-entropy + softmax w.r.t. pre-softmax logits
            # dL/d_combined[j] = sm[j] - one_hot[j]
            d_combined = [sm[j] - one_hot[j] for j in range(self.output_size)]

            # For each active expert: d_loss/d_W2 = weight * outer(d_combined, h)
            for idx, w in zip(indices, weights):
                exp = self.experts[idx]
                # dL/d_W2[i][j] = w * d_combined[i] * h[j]
                for i in range(self.output_size):
                    for j in range(len(exp.last_h)):
                        exp.W2[i][j] -= lr * w * d_combined[i] * exp.last_h[j]
                # dL/d_b2[i] = w * d_combined[i]
                for i in range(self.output_size):
                    exp.b2[i] -= lr * w * d_combined[i]

        # ---- numerical gradient for router (lightweight, low-lr) -------
        router_lr = lr * 0.1
        eps = 1e-4

        def _seq_loss():
            """Compute loss with current router weights."""
            loss = 0.0
            for x, target_idx in zip(input_seq, targets):
                p = self.router.forward(x)
                idxs, ws = self.router.select_top_k(p, self.top_k)
                comb = _zeros(self.output_size)
                for ei, wt in zip(idxs, ws):
                    comb = _vec_add(comb, _vec_scale(self.experts[ei].forward(x), wt))
                sm = softmax(comb)
                oh = [1.0 if i == target_idx else 0.0 for i in range(self.output_size)]
                loss += cross_entropy_loss(sm, oh)
            return loss / len(input_seq)

        for row_i in range(len(self.router.W)):
            for col_j in range(len(self.router.W[0])):
                orig = self.router.W[row_i][col_j]
                self.router.W[row_i][col_j] = orig + eps
                loss_plus = _seq_loss()
                self.router.W[row_i][col_j] = orig - eps
                loss_minus = _seq_loss()
                self.router.W[row_i][col_j] = orig
                grad = (loss_plus - loss_minus) / (2 * eps)
                self.router.W[row_i][col_j] -= router_lr * grad

        return total_loss / n_steps

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
        preds = []
        for x in input_seq:
            combined, _ = self.forward(x)
            probs = softmax(combined)
            preds.append(probs.index(max(probs)))
        return preds

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def get_params_count(self):
        """Total number of scalar parameters (all experts + router)."""
        total = self.router.param_count
        for exp in self.experts:
            total += exp.param_count
        return total

    def get_active_params_count(self):
        """Parameters used during a single inference (top_k experts + router)."""
        active = self.router.param_count
        for exp in self.experts[:self.top_k]:
            active += exp.param_count
        return active


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 2 — Mixture of Experts Demo")
    print("=" * 60)

    # Tiny toy task: 8-dim one-hot input, 4 classes, 4 experts, top-2
    INPUT_SIZE = 8
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4
    NUM_EXPERTS = 4
    TOP_K = 2

    model = MoEModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                     num_experts=NUM_EXPERTS, top_k=TOP_K, seed=0)

    print(f"\nTotal params   : {model.get_params_count()}")
    print(f"Active params  : {model.get_active_params_count()}")
    print(f"Sparsity ratio : {model.get_active_params_count() / model.get_params_count():.2%}")

    # Build a tiny sequence
    rng = random.Random(7)
    seq = [[1.0 if i == rng.randint(0, INPUT_SIZE - 1) else 0.0
            for i in range(INPUT_SIZE)] for _ in range(5)]
    targets = [rng.randint(0, OUTPUT_SIZE - 1) for _ in range(5)]

    print("\n--- Forward pass ---")
    results = model.forward_sequence(seq)
    for step, (out, (idxs, wts)) in enumerate(results):
        print(f"  Step {step}: experts={idxs}, weights={[f'{w:.3f}' for w in wts]}, "
              f"output_max={max(out):.4f}")

    print("\n--- Training (5 steps) ---")
    for epoch in range(5):
        loss = model.train_step(seq, targets, lr=0.05)
        print(f"  Epoch {epoch + 1}: loss={loss:.4f}")

    print("\n--- Predictions ---")
    preds = model.predict(seq)
    print(f"  Predicted : {preds}")
    print(f"  Targets   : {targets}")
    print("\nDone.")
