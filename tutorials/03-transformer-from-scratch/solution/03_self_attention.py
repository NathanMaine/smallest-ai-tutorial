"""
Chapter 3 — Self-Attention
===========================

THE core innovation of the Transformer architecture.

What this module teaches
-------------------------
Self-attention allows every position in a sequence to look at every other
position and decide how much to "pay attention" to it. This is how a
Transformer understands relationships between words — regardless of distance.

The mechanism uses three learned projections per position:
  - **Query (Q):** "What am I looking for?"
  - **Key (K):**   "What do I contain?"
  - **Value (V):** "What information do I provide?"

Attention score between positions i and j = dot(Q[i], K[j]) / sqrt(d_k).
After softmax normalisation, these scores become weights that blend the
value vectors into a context-aware output.

Causal masking prevents attending to future positions — essential for
autoregressive language models (predicting the next token).

Builds on:
  - 01_math_foundations.py (dot_product, transpose) from level-a-abcs
  - 04_loss_function.py (softmax) from level-a-abcs
  - 01_embeddings.py, 02_positional_encoding.py from this level

Chapter roadmap (Level C — The Reader)
---------------------------------------
  Chapter 1:  Embeddings — words to vectors
  Chapter 2:  Positional encoding — injecting word order
  Chapter 3:  Self-attention — scaled dot-product attention  ← you are here
  Chapter 4:  Multi-head attention — parallel attention heads
"""

import importlib
import sys
import os
import math
import random

# ---------------------------------------------------------------------------
# Import helpers from Level A
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product
transpose = math_fn.transpose

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def create_causal_mask(seq_len):
    """Create a causal (look-ahead) mask for autoregressive decoding.

    Returns a seq_len x seq_len boolean matrix where mask[i][j] = True
    if j > i, meaning position i should NOT attend to position j (a future
    position).

    The diagonal and below are False (allowed), everything above is True
    (blocked). This ensures that when predicting token at position i, the
    model can only see tokens at positions 0 through i.

    Parameters
    ----------
    seq_len : int — length of the sequence

    Returns
    -------
    list[list[bool]] — seq_len x seq_len boolean mask
    """
    return [[j > i for j in range(seq_len)] for i in range(seq_len)]


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    This is the core attention computation from "Attention Is All You Need":

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    For each query position i:
      1. Compute score[i][j] = dot(Q[i], K[j]) / sqrt(d_k) for all j
      2. If mask is provided and mask[i][j] is True, set score to -1e9
      3. Apply softmax to get attention weights
      4. Compute output as weighted sum of value vectors

    Parameters
    ----------
    Q : list[list[float]] — query vectors [seq_len_q x d_k]
    K : list[list[float]] — key vectors [seq_len_k x d_k]
    V : list[list[float]] — value vectors [seq_len_k x d_v]
    mask : list[list[bool]] or None — True at positions to block

    Returns
    -------
    (outputs, attention_weights)
        outputs : list[list[float]] — [seq_len_q x d_v]
        attention_weights : list[list[float]] — [seq_len_q x seq_len_k]
    """
    d_k = len(Q[0])
    scale = math.sqrt(d_k)
    seq_len_q = len(Q)
    seq_len_k = len(K)

    # Compute scaled scores
    scores = []
    for i in range(seq_len_q):
        row = []
        for j in range(seq_len_k):
            score = dot_product(Q[i], K[j]) / scale
            if mask is not None and mask[i][j]:
                score = -1e9
            row.append(score)
        scores.append(row)

    # Apply softmax to each row
    attention_weights = [softmax(row) for row in scores]

    # Compute weighted sum of values
    d_v = len(V[0])
    outputs = []
    for i in range(seq_len_q):
        out_vec = [0.0] * d_v
        for j in range(seq_len_k):
            w = attention_weights[i][j]
            for d in range(d_v):
                out_vec[d] += w * V[j][d]
        outputs.append(out_vec)

    return outputs, attention_weights


# ---------------------------------------------------------------------------
# Self-Attention class (single head)
# ---------------------------------------------------------------------------

class SelfAttention:
    """A single attention head that projects inputs to Q, K, V and applies
    scaled dot-product attention.

    The projection matrices W_q, W_k, W_v each have shape [head_dim x embed_dim].
    Given input x of shape [seq_len x embed_dim], the forward pass computes:

        Q = x @ W_q^T   (each position projected to a query vector)
        K = x @ W_k^T   (each position projected to a key vector)
        V = x @ W_v^T   (each position projected to a value vector)

    Then applies scaled_dot_product_attention(Q, K, V, mask).

    Parameters
    ----------
    embed_dim : int — dimension of input vectors
    head_dim  : int — dimension of Q, K, V vectors (often embed_dim // num_heads)
    seed      : int — random seed for reproducible Xavier initialisation
    """

    def __init__(self, embed_dim, head_dim, seed=42):
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        rng = random.Random(seed)
        limit = math.sqrt(6.0 / (embed_dim + head_dim))

        # W_q, W_k, W_v each [head_dim x embed_dim]
        self.W_q = [[rng.uniform(-limit, limit) for _ in range(embed_dim)]
                     for _ in range(head_dim)]
        self.W_k = [[rng.uniform(-limit, limit) for _ in range(embed_dim)]
                     for _ in range(head_dim)]
        self.W_v = [[rng.uniform(-limit, limit) for _ in range(embed_dim)]
                     for _ in range(head_dim)]

        # Stored for potential backprop
        self.Q = None
        self.K = None
        self.V = None
        self.attn_weights = None

    def forward(self, x, mask=None):
        """Project input to Q, K, V and apply attention.

        Parameters
        ----------
        x    : list[list[float]] — [seq_len x embed_dim] input vectors
        mask : list[list[bool]] or None — causal mask

        Returns
        -------
        (outputs, attn_weights)
            outputs : list[list[float]] — [seq_len x head_dim]
            attn_weights : list[list[float]] — [seq_len x seq_len]
        """
        seq_len = len(x)

        # Project: Q[i] = W_q @ x[i], etc.
        self.Q = [self._mat_vec(self.W_q, x[i]) for i in range(seq_len)]
        self.K = [self._mat_vec(self.W_k, x[i]) for i in range(seq_len)]
        self.V = [self._mat_vec(self.W_v, x[i]) for i in range(seq_len)]

        outputs, self.attn_weights = scaled_dot_product_attention(
            self.Q, self.K, self.V, mask=mask
        )
        return outputs, self.attn_weights

    @staticmethod
    def _mat_vec(matrix, vector):
        """Multiply matrix [rows x cols] by vector [cols] -> [rows]."""
        return [dot_product(row, vector) for row in matrix]


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 phase1-from-scratch/level-c-reader/03_self_attention.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3 — Self-Attention Demo")
    print("Scaled dot-product attention from scratch")
    print("=" * 60)

    # --- Demo 1: Basic attention ---
    print("\n--- Scaled Dot-Product Attention ---")
    Q = [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]]
    K = [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]]
    V = [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]]

    outputs, weights = scaled_dot_product_attention(Q, K, V)
    print("Q (identity-like queries):")
    for row in Q:
        print(f"  {row}")
    print("Attention weights (each row sums to 1):")
    for i, row in enumerate(weights):
        print(f"  pos {i}: [{', '.join(f'{w:.4f}' for w in row)}]")
    print("Outputs (weighted blend of V):")
    for i, vec in enumerate(outputs):
        print(f"  pos {i}: [{', '.join(f'{v:.4f}' for v in vec)}]")

    # --- Demo 2: Causal masking ---
    print("\n--- Causal Mask ---")
    mask = create_causal_mask(4)
    print("4x4 causal mask (True = blocked):")
    for i, row in enumerate(mask):
        print(f"  pos {i}: {row}")

    print("\nAttention with causal mask:")
    Q2 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    K2 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V2 = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    mask3 = create_causal_mask(3)
    out_masked, w_masked = scaled_dot_product_attention(Q2, K2, V2, mask=mask3)
    print("Masked attention weights:")
    for i, row in enumerate(w_masked):
        print(f"  pos {i}: [{', '.join(f'{w:.4f}' for w in row)}]")
    print("Position 0 attends 100% to itself (can't see future).")

    # --- Demo 3: SelfAttention class ---
    print("\n--- SelfAttention Class ---")
    sa = SelfAttention(embed_dim=8, head_dim=4, seed=42)
    x = [[0.1 * (i + j) for j in range(8)] for i in range(3)]
    print(f"Input: {len(x)} vectors of dim {len(x[0])}")
    sa_out, sa_w = sa.forward(x)
    print(f"Output: {len(sa_out)} vectors of dim {len(sa_out[0])}")
    print("Attention weights:")
    for i, row in enumerate(sa_w):
        print(f"  pos {i}: [{', '.join(f'{w:.4f}' for w in row)}]")

    print("\n" + "=" * 60)
    print("Chapter 3 complete. Chapter 4: Multi-Head Attention.")
    print("=" * 60)
