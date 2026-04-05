"""
Chapter 4 — Multi-Head Attention
==================================

Multiple heads attend to different aspects of the input simultaneously.

What this module teaches
-------------------------
A single attention head can only focus on one type of relationship at a time.
Multi-head attention runs several attention heads in parallel — each with its
own learned Q, K, V projections — so the model can simultaneously attend to:

  - **Syntactic patterns** — subject-verb agreement across distance
  - **Semantic similarity** — words with related meanings
  - **Positional proximity** — nearby words in the sequence
  - **Coreference** — pronouns and their antecedents

Each head produces a [seq_len x head_dim] output. These are concatenated back
to [seq_len x embed_dim] and projected through a final output matrix to mix
information across heads.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_output
    where head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)

Builds on:
  - 01_math_foundations.py (dot_product) from level-a-abcs
  - 03_self_attention.py (SelfAttention) from this level

Chapter roadmap (Level C — The Reader)
---------------------------------------
  Chapter 1:  Embeddings — words to vectors
  Chapter 2:  Positional encoding — injecting word order
  Chapter 3:  Self-attention — scaled dot-product attention
  Chapter 4:  Multi-head attention — parallel attention heads  ← you are here
"""

import importlib
import sys
import os
import random
import math

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(__file__))

math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product

attn_mod = importlib.import_module('03_self_attention')
SelfAttention = attn_mod.SelfAttention


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    """Multi-head attention: run multiple SelfAttention heads in parallel,
    concatenate their outputs, and project through a final linear layer.

    Parameters
    ----------
    embed_dim : int — total embedding dimension (must be divisible by num_heads)
    num_heads : int — number of parallel attention heads
    seed      : int — random seed for reproducible initialisation
    """

    def __init__(self, embed_dim, num_heads, seed=42):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Create independent attention heads, each with a different seed
        # so they learn different projections
        self.heads = [
            SelfAttention(embed_dim, self.head_dim, seed=seed + i)
            for i in range(num_heads)
        ]

        # Output projection W_output [embed_dim x embed_dim]
        rng = random.Random(seed + num_heads)
        limit = math.sqrt(6.0 / (embed_dim + embed_dim))
        self.W_output = [
            [rng.uniform(-limit, limit) for _ in range(embed_dim)]
            for _ in range(embed_dim)
        ]

    def forward(self, x, mask=None):
        """Run all heads in parallel, concatenate, and project.

        Parameters
        ----------
        x    : list[list[float]] — [seq_len x embed_dim] input vectors
        mask : list[list[bool]] or None — causal mask

        Returns
        -------
        list[list[float]] — [seq_len x embed_dim] output vectors
        """
        seq_len = len(x)

        # 1. Run each head
        head_outputs = []
        for head in self.heads:
            out, _ = head.forward(x, mask=mask)
            head_outputs.append(out)  # each is [seq_len x head_dim]

        # 2. Concatenate heads: for each position, concat all head outputs
        concat = []
        for pos in range(seq_len):
            cat_vec = []
            for h in range(self.num_heads):
                cat_vec.extend(head_outputs[h][pos])
            concat.append(cat_vec)  # [embed_dim]

        # 3. Project through W_output
        output = []
        for pos in range(seq_len):
            out_vec = [dot_product(self.W_output[d], concat[pos])
                       for d in range(self.embed_dim)]
            output.append(out_vec)

        return output


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 phase1-from-scratch/level-c-reader/04_multi_head_attention.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4 — Multi-Head Attention Demo")
    print("Parallel attention heads from scratch")
    print("=" * 60)

    embed_dim = 16
    num_heads = 4
    seq_len = 5

    print(f"\nConfiguration: embed_dim={embed_dim}, num_heads={num_heads}, "
          f"head_dim={embed_dim // num_heads}")

    # Create sample input: 5 vectors of dim 16
    random.seed(99)
    x = [[random.gauss(0, 0.5) for _ in range(embed_dim)]
         for _ in range(seq_len)]

    mha = MultiHeadAttention(embed_dim, num_heads, seed=42)
    print(f"Input:  {seq_len} vectors of dim {embed_dim}")
    print(f"Heads:  {len(mha.heads)}")

    output = mha.forward(x)
    print(f"Output: {len(output)} vectors of dim {len(output[0])}")

    print("\nOutput vectors (first 4 dims of each):")
    for i, vec in enumerate(output):
        snippet = ', '.join(f'{v:.4f}' for v in vec[:4])
        print(f"  pos {i}: [{snippet}, ...]")

    # Show that each head attends differently
    print("\nPer-head attention weights for position 2:")
    for h, head in enumerate(mha.heads):
        w = head.attn_weights[2]
        print(f"  head {h}: [{', '.join(f'{v:.4f}' for v in w)}]")
    print("(Different heads attend to different positions.)")

    # Demo with causal mask
    from importlib import import_module
    attn = import_module('03_self_attention')
    mask = attn.create_causal_mask(seq_len)
    output_masked = mha.forward(x, mask=mask)
    print(f"\nWith causal mask: {len(output_masked)} vectors of dim {len(output_masked[0])}")
    print("Position 0 head weights (should only attend to itself):")
    for h, head in enumerate(mha.heads):
        w = head.attn_weights[0]
        print(f"  head {h}: [{', '.join(f'{v:.4f}' for v in w)}]")

    print("\n" + "=" * 60)
    print("Chapter 4 complete. Next: Feed-Forward Network.")
    print("=" * 60)
