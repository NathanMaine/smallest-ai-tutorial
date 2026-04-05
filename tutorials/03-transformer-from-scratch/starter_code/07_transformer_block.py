"""
Chapter 7 — Transformer Block
==============================

A single Transformer block is the repeating unit of the architecture.
It combines multi-head self-attention and a feed-forward network, each
wrapped with layer normalization and a residual connection.

This uses the Pre-Norm variant (Pre-LN), where layer normalization is
applied *before* each sub-layer rather than after. Pre-Norm is preferred
in modern Transformers (GPT-2, LLaMA) because it stabilizes training.

Pre-Norm pattern:
    norm_x    = norm1(x)
    x         = x + Attention(norm_x)    # residual
    norm_x    = norm2(x)
    x         = x + FFN(norm_x)          # residual
    return x

This block gets stacked N times in Chapter 8 to form a complete model.

Builds on: 04, 05, 06 (MultiHeadAttention, FeedForward, LayerNorm)
"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mha_mod = importlib.import_module('04_multi_head_attention')
ffn_mod = importlib.import_module('05_feed_forward')
ln_mod  = importlib.import_module('06_layer_norm_residual')

MultiHeadAttention = mha_mod.MultiHeadAttention
FeedForward        = ffn_mod.FeedForward
LayerNorm          = ln_mod.LayerNorm
residual_add       = ln_mod.residual_add


class TransformerBlock:
    """One complete Transformer layer: Pre-Norm attention + Pre-Norm FFN.

    Parameters
    ----------
    embed_dim : int       — token embedding dimensionality
    num_heads : int       — number of parallel attention heads
    ff_dim    : int|None  — inner dim of the FFN (default: 4 * embed_dim)
    seed      : int       — base random seed
    """

    def __init__(self, embed_dim, num_heads, ff_dim=None, seed=42):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(embed_dim, num_heads, seed)
        self.ffn       = FeedForward(embed_dim, ff_dim, seed + 1)
        self.norm1     = LayerNorm(embed_dim)
        self.norm2     = LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """Run one Transformer block: norm → attention → residual → norm → FFN → residual.

        Pre-Norm pattern:
          1. norm_x   = self.norm1 applied to each vector in x
          2. attn_out = self.attention.forward(norm_x, mask)
          3. x        = residual_add(x, attn_out)  — add attention output to input
          4. norm_x   = self.norm2 applied to each vector in x
          5. ffn_out  = self.ffn applied to each vector in norm_x
          6. x        = residual_add(x, ffn_out)   — add FFN output
          7. return x

        Parameters
        ----------
        x    : list[list[float]] — [seq_len x embed_dim]
        mask : list[list[bool]] or None

        Returns
        -------
        list[list[float]] — [seq_len x embed_dim], same shape as input
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Apply the 7-step Pre-Norm pattern above.\n"
            "  Hint: norm1/norm2 are LayerNorm objects — call norm.normalize(vec) for each vector.\n"
            "  self.attention.forward(norm_x, mask) returns (outputs, weights) — you only need outputs.\n"
            "  self.ffn.forward(vec) transforms one vector at a time.\n"
            "  residual_add(a, b) adds two equal-length vectors element-wise."
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    block = TransformerBlock(embed_dim=16, num_heads=4, seed=42)
    x = [[0.1 * (i + j) for j in range(16)] for i in range(5)]

    output = block.forward(x)
    print(f"Input shape:  {len(x)} x {len(x[0])}")
    print(f"Output shape: {len(output)} x {len(output[0])}")
    assert len(output) == len(x), "Output should have same seq_len"
    assert len(output[0]) == len(x[0]), "Output should have same embed_dim"
    print("TransformerBlock.forward: PASS")
