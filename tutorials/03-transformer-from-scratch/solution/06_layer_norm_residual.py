"""
Chapter 6 — Layer Normalization + Residual Connections
=======================================================

What this module teaches
-------------------------
Two techniques that make deep Transformers trainable: Layer Normalization
and Residual (skip) Connections. Together they form the "Add & Norm" step
that wraps every sub-layer (attention and FFN) in the Transformer.

Layer Normalization
--------------------
Layer Norm standardises each vector *independently* to have mean ≈ 0 and
variance ≈ 1, then rescales with learned parameters gamma (scale) and beta
(shift):

    mean   = sum(x) / len(x)
    var    = sum((xi - mean)^2) / len(x)
    x_norm = (x - mean) / sqrt(var + eps)
    output = gamma * x_norm + beta

Key differences from Batch Norm:
  - Operates on a single sample (one vector), not across the batch.
  - Stable for variable-length sequences and small batches.
  - Standard in NLP and Transformers.

Residual Connections
---------------------
Inspired by ResNet, residual connections add the *input* of a sub-layer
directly to its *output*:

    output = x + SubLayer(x)

This provides:
  1. A "gradient highway" — gradients flow directly through the addition,
     bypassing the sub-layer. Prevents vanishing gradients in deep networks.
  2. An identity shortcut — if the sub-layer learns to output near-zero,
     the block effectively becomes an identity function. Layers can be
     added without hurting performance.

Full "Add & Norm" pattern used in every Transformer block:
    y = LayerNorm(x + Attention(x))
    z = LayerNorm(y + FFN(y))

Implementation notes
---------------------
eps (default 1e-5) is added inside the sqrt to prevent division by zero
when variance is exactly 0 (e.g., constant input vector).
"""

import math


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm:
    """Layer Normalization over a single vector.

    Parameters
    ----------
    dim : int   — length of the vectors to normalise
    eps : float — small constant added to variance for numerical stability
    """

    def __init__(self, dim, eps=1e-5):
        self.dim   = dim
        self.eps   = eps
        self.gamma = [1.0] * dim  # learnable scale, initialised to 1
        self.beta  = [0.0] * dim  # learnable shift, initialised to 0

    def forward(self, x):
        """Normalise a single vector.

        Parameters
        ----------
        x : list of floats — length dim

        Returns
        -------
        list of floats — normalised vector, same length as x
        """
        n    = len(x)
        mean = sum(x) / n
        var  = sum((xi - mean) ** 2 for xi in x) / n
        std  = math.sqrt(var + self.eps)

        return [
            self.gamma[i] * (x[i] - mean) / std + self.beta[i]
            for i in range(n)
        ]

    def forward_sequence(self, sequence):
        """Normalise every vector in a sequence.

        Parameters
        ----------
        sequence : list of lists — shape [seq_len x dim]

        Returns
        -------
        list of lists — shape [seq_len x dim], each vector independently normalised
        """
        return [self.forward(vec) for vec in sequence]


# ---------------------------------------------------------------------------
# Residual addition
# ---------------------------------------------------------------------------

def residual_add(x, sublayer_output):
    """Element-wise addition of a residual connection.

    Implements: output[i] = x[i] + sublayer_output[i]  (for each position i,
    and element-wise within each vector).

    Parameters
    ----------
    x               : list of lists — shape [seq_len x dim], original input
    sublayer_output : list of lists — shape [seq_len x dim], sub-layer result

    Returns
    -------
    list of lists — shape [seq_len x dim]
    """
    return [
        [xi + si for xi, si in zip(x_vec, s_vec)]
        for x_vec, s_vec in zip(x, sublayer_output)
    ]


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 "phase1-from-scratch/level-c-reader/06_layer_norm_residual.py"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 6 — Layer Norm + Residual Demo")
    print("Pure Python, no libraries")
    print("=" * 60)

    # --- Layer Norm on a single vector ---
    print("\n--- Layer Normalization (single vector) ---")
    dim = 8
    ln  = LayerNorm(dim)

    x_raw = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    x_norm = ln.forward(x_raw)

    mean_raw  = sum(x_raw) / len(x_raw)
    mean_norm = sum(x_norm) / len(x_norm)
    var_norm  = sum((v - mean_norm) ** 2 for v in x_norm) / len(x_norm)

    print(f"Input   : {[f'{v:.1f}' for v in x_raw]}")
    print(f"Output  : {[f'{v:.4f}' for v in x_norm]}")
    print(f"Mean before : {mean_raw:.4f}  →  after : {mean_norm:.2e}  (≈ 0)")
    print(f"Var  before : computed  →  after : {var_norm:.6f}  (≈ 1)")

    # --- Layer Norm on a sequence ---
    print("\n--- Layer Normalization (sequence) ---")
    seq = [
        [float(i + j * 2) for i in range(dim)]
        for j in range(3)
    ]
    seq_norm = ln.forward_sequence(seq)
    print(f"Sequence shape : {len(seq)} × {len(seq[0])}")
    print(f"Normalised shape: {len(seq_norm)} × {len(seq_norm[0])}")
    for pos, vec in enumerate(seq_norm):
        m = sum(vec) / len(vec)
        print(f"  position {pos}: mean = {m:.2e}  (first 3: {[f'{v:.3f}' for v in vec[:3]]})")

    # --- Residual add ---
    print("\n--- Residual Connection ---")
    x = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    sub = [[0.1, -0.2, 0.3, -0.4], [-0.1, 0.2, -0.3, 0.4]]
    out = residual_add(x, sub)
    print(f"x         : {x}")
    print(f"sublayer  : {sub}")
    print(f"x + sub   : {[[round(v, 2) for v in row] for row in out]}")

    # --- Full "Add & Norm" pattern ---
    print("\n--- Full 'Add & Norm' (residual + layer norm) ---")
    dim2 = 4
    ln2  = LayerNorm(dim2)
    orig = [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]]
    fake_attention = [[0.5, -0.3, 0.2, -0.1], [0.1, 0.4, -0.2, 0.3]]

    added = residual_add(orig, fake_attention)
    normed = ln2.forward_sequence(added)
    print(f"After Add & Norm (shape {len(normed)} × {len(normed[0])}):")
    for pos, vec in enumerate(normed):
        print(f"  position {pos}: {[f'{v:.4f}' for v in vec]}")

    print("\n" + "=" * 60)
    print("Layer Norm + Residual complete. Transformer block ready.")
    print("=" * 60)
