"""
Chapter 3 — Self-Attention
===========================

THE core innovation of the Transformer architecture.

Self-attention allows every position in a sequence to look at every other
position and decide how much to "pay attention" to it.

The mechanism uses three learned projections per position:
  - Query (Q): "What am I looking for?"
  - Key   (K): "What do I contain?"
  - Value (V): "What information do I provide?"

Attention score between positions i and j:
    score(i, j) = dot(Q[i], K[j]) / sqrt(d_k)

After softmax normalisation, scores become weights that blend value vectors:
    output[i] = sum(attention_weight[i][j] * V[j] for all j)

Why divide by sqrt(d_k)?
  The dot products grow in magnitude as d_k increases, pushing softmax into
  regions where gradients vanish. Dividing by sqrt(d_k) keeps the scores
  in a reasonable range regardless of embedding dimension.

Causal masking:
  For language models (predicting the next token), position i must not attend
  to position j > i (a future token). The mask sets those scores to -1e9
  before softmax, making their attention weights effectively zero.

Builds on: 01-mlp-from-scratch (dot_product, transpose, softmax)
"""

import importlib
import sys
import os
import math
import random

# Import from Tutorial 01
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
math_fn  = importlib.import_module('01_math_foundations')
loss_mod = importlib.import_module('04_loss_function')

dot_product = math_fn.dot_product
transpose   = math_fn.transpose
softmax     = loss_mod.softmax


def create_causal_mask(seq_len):
    """Create a causal (look-ahead) mask.

    Returns a seq_len x seq_len boolean matrix where mask[i][j] = True
    if j > i (position i should NOT attend to future position j).

    The diagonal and below are False (allowed). Everything above is True (blocked).

    Parameters
    ----------
    seq_len : int

    Returns
    -------
    list[list[bool]] — seq_len x seq_len
    """
    raise NotImplementedError(
        "Your turn! mask[i][j] = True if j > i else False"
    )


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    For each query position i:
      1. Compute score[i][j] = dot(Q[i], K[j]) / sqrt(d_k) for all j
      2. If mask[i][j] is True, set score to -1e9 (effectively blocks it)
      3. Apply softmax to get attention weights (each row sums to 1)
      4. Compute output as weighted sum of value vectors

    Parameters
    ----------
    Q : list[list[float]] — query vectors [seq_len_q x d_k]
    K : list[list[float]] — key vectors   [seq_len_k x d_k]
    V : list[list[float]] — value vectors [seq_len_k x d_v]
    mask : list[list[bool]] or None — True at positions to block

    Returns
    -------
    (outputs, attention_weights)
        outputs          : list[list[float]] — [seq_len_q x d_v]
        attention_weights: list[list[float]] — [seq_len_q x seq_len_k]
    """
    raise NotImplementedError(
        "Your turn!\n"
        "  d_k = len(Q[0])\n"
        "  scale = math.sqrt(d_k)\n"
        "  For each i (query) and j (key):\n"
        "    score = dot_product(Q[i], K[j]) / scale\n"
        "    if mask is not None and mask[i][j]: score = -1e9\n"
        "  Apply softmax to each row of scores → attention_weights\n"
        "  output[i] = weighted sum of V vectors using attention_weights[i]"
    )


class SelfAttention:
    """A single attention head: projects inputs to Q, K, V and applies attention.

    Given input x of shape [seq_len x embed_dim]:
        Q = x @ W_q^T   (project each position to a query vector)
        K = x @ W_k^T   (project each position to a key vector)
        V = x @ W_v^T   (project each position to a value vector)
    Then: output = scaled_dot_product_attention(Q, K, V, mask)

    Parameters
    ----------
    embed_dim : int — dimension of input vectors
    head_dim  : int — dimension of Q, K, V vectors
    seed      : int — random seed for Xavier initialisation
    """

    def __init__(self, embed_dim, head_dim, seed=42):
        self.embed_dim = embed_dim
        self.head_dim  = head_dim

        rng   = random.Random(seed)
        limit = math.sqrt(6.0 / (embed_dim + head_dim))

        # W_q, W_k, W_v each [head_dim x embed_dim]
        self.W_q = [[rng.uniform(-limit, limit) for _ in range(embed_dim)] for _ in range(head_dim)]
        self.W_k = [[rng.uniform(-limit, limit) for _ in range(embed_dim)] for _ in range(head_dim)]
        self.W_v = [[rng.uniform(-limit, limit) for _ in range(embed_dim)] for _ in range(head_dim)]

        self.Q = self.K = self.V = self.attn_weights = None

    def forward(self, x, mask=None):
        """Project input to Q, K, V and apply attention.

        Parameters
        ----------
        x    : list[list[float]] — [seq_len x embed_dim]
        mask : list[list[bool]] or None

        Returns
        -------
        (outputs, attn_weights)
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  For each position i, compute:\n"
            "    Q[i] = W_q @ x[i]  (dot each row of W_q with x[i])\n"
            "    K[i] = W_k @ x[i]\n"
            "    V[i] = W_v @ x[i]\n"
            "  Then call scaled_dot_product_attention(Q, K, V, mask)"
        )

    @staticmethod
    def _mat_vec(matrix, vector):
        """Multiply matrix [rows x cols] by vector [cols] → [rows]."""
        return [dot_product(row, vector) for row in matrix]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test: 3 positions, 4-dim Q/K/V
    Q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    K = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    V = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]

    outputs, weights = scaled_dot_product_attention(Q, K, V)
    print("Attention weights (each row should sum to 1):")
    for i, row in enumerate(weights):
        print(f"  pos {i}: {[round(w, 4) for w in row]}, sum={round(sum(row), 6)}")

    # Test causal mask
    mask = create_causal_mask(4)
    print("\n4x4 causal mask (True = blocked, shouldn't see future):")
    for i, row in enumerate(mask):
        print(f"  pos {i}: {row}")
