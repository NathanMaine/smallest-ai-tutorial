"""
Chapter 5 — Feed-Forward Network
=================================

What this module teaches
-------------------------
The Transformer's Feed-Forward Network (FFN) is the second major sub-layer in
every Transformer block. After self-attention mixes information *across*
positions, the FFN processes each position *independently*, applying the same
two-layer MLP to every token vector.

Architecture
------------
FFN(x) = W2 * ReLU(W1 * x + b1) + b2

  - W1 : [ff_dim  x embed_dim]  — expands to a higher-dimensional space
  - b1 : [ff_dim]
  - W2 : [embed_dim x ff_dim]   — projects back to embed_dim
  - b2 : [embed_dim]

The expansion ratio ff_dim / embed_dim is typically 4 (default here).
ReLU introduces the non-linearity that lets the network learn complex mappings.

Why position-wise?
------------------
Each token is transformed identically and independently — the FFN has no
concept of position or sequence order. This is intentional: attention already
captures positional relationships; the FFN's job is to transform the *content*
of each position's representation.

Xavier Initialisation
---------------------
Weights are drawn uniformly from [-limit, +limit] where:
    limit = sqrt(6 / (fan_in + fan_out))
This keeps activations and gradients in a reasonable range at the start
of training, preventing vanishing/exploding signals.

Builds on
---------
  - level-a-abcs/01_math_foundations.py  (dot_product, vector_add)
  - level-a-abcs/02_single_neuron.py     (relu)
"""

import importlib
import sys
import os
import random

# ---------------------------------------------------------------------------
# Import helpers from Level A
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product
vector_add  = math_fn.vector_add
neuron_mod  = importlib.import_module('02_single_neuron')
relu        = neuron_mod.relu


# ---------------------------------------------------------------------------
# FeedForward class
# ---------------------------------------------------------------------------

class FeedForward:
    """Position-wise Feed-Forward Network.

    Applies a two-layer MLP with ReLU activation to every position in the
    sequence independently (i.e., weights are shared across positions but
    each position is processed separately).

    Parameters
    ----------
    embed_dim : int  — input and output dimensionality (d_model)
    ff_dim    : int  — inner (hidden) dimensionality, defaults to 4 * embed_dim
    seed      : int  — random seed for reproducible weight initialisation
    """

    def __init__(self, embed_dim, ff_dim=None, seed=42):
        self.embed_dim = embed_dim
        self.ff_dim    = ff_dim if ff_dim is not None else 4 * embed_dim

        rng = random.Random(seed)

        # Xavier uniform init for W1: shape [ff_dim x embed_dim]
        limit_w1 = (6.0 / (self.embed_dim + self.ff_dim)) ** 0.5
        self.W1 = [
            [rng.uniform(-limit_w1, limit_w1) for _ in range(self.embed_dim)]
            for _ in range(self.ff_dim)
        ]
        self.b1 = [0.0] * self.ff_dim

        # Xavier uniform init for W2: shape [embed_dim x ff_dim]
        limit_w2 = (6.0 / (self.ff_dim + self.embed_dim)) ** 0.5
        self.W2 = [
            [rng.uniform(-limit_w2, limit_w2) for _ in range(self.ff_dim)]
            for _ in range(self.embed_dim)
        ]
        self.b2 = [0.0] * self.embed_dim

        # Intermediates for potential backprop
        self._last_inputs  = None
        self._last_hiddens = None
        self._last_outputs = None

    def _linear(self, W, b, x):
        """Compute W @ x + b.

        W : list of lists — shape [out_dim x in_dim]
        b : list          — shape [out_dim]
        x : list          — shape [in_dim]

        Returns list of length out_dim.
        """
        return vector_add(
            [dot_product(row, x) for row in W],
            b
        )

    def forward(self, x):
        """Apply the FFN to every position in the sequence.

        Parameters
        ----------
        x : list of lists — shape [seq_len x embed_dim]

        Returns
        -------
        list of lists — shape [seq_len x embed_dim]
        """
        outputs = []
        hiddens = []

        for vec in x:
            # First linear layer + ReLU: hidden = relu(W1 @ vec + b1)
            pre_hidden = self._linear(self.W1, self.b1, vec)
            hidden     = [relu(h) for h in pre_hidden]

            # Second linear layer: out = W2 @ hidden + b2
            out = self._linear(self.W2, self.b2, hidden)

            hiddens.append(hidden)
            outputs.append(out)

        # Store intermediates
        self._last_inputs  = x
        self._last_hiddens = hiddens
        self._last_outputs = outputs

        return outputs


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 "phase1-from-scratch/level-c-reader/05_feed_forward.py"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 5 — Feed-Forward Network Demo")
    print("Pure Python, no libraries")
    print("=" * 60)

    embed_dim = 8
    seq_len   = 3
    ff = FeedForward(embed_dim, seed=42)

    print(f"\nEmbed dim : {ff.embed_dim}")
    print(f"FF dim    : {ff.ff_dim}  (= 4 × embed_dim)")
    print(f"W1 shape  : {len(ff.W1)} × {len(ff.W1[0])}")
    print(f"W2 shape  : {len(ff.W2)} × {len(ff.W2[0])}")

    # Build a tiny input: 3 positions, each an 8-dim vector
    x = [[float(i + j * 0.5) for i in range(embed_dim)] for j in range(seq_len)]
    print(f"\nInput shape : {len(x)} × {len(x[0])}")

    out = ff.forward(x)
    print(f"Output shape: {len(out)} × {len(out[0])}")

    print("\nSample outputs (first 4 values per position):")
    for pos, vec in enumerate(out):
        preview = [f"{v:.4f}" for v in vec[:4]]
        print(f"  position {pos}: [{', '.join(preview)}, ...]")

    # Verify position-independence
    x_mod = [list(v) for v in x]
    x_mod[0] = [99.0] * embed_dim
    out_mod = ff.forward(x_mod)
    print("\nAfter changing position 0 input to all-99:")
    print(f"  position 0 changed : {out[0][:2]} → {out_mod[0][:2]}")
    print(f"  position 1 unchanged: {out[1][:2]} == {out_mod[1][:2]}  "
          f"{'OK' if out[1] == out_mod[1] else 'FAIL'}")

    print("\n" + "=" * 60)
    print("FFN complete. Chapter 6: Layer Norm + Residual.")
    print("=" * 60)
