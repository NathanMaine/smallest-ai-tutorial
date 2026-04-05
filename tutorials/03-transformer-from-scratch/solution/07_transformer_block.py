"""
Chapter 7 — Transformer Block
==============================

What this module teaches
-------------------------
A single Transformer block is the repeating unit of the Transformer
architecture. It combines multi-head self-attention and a feed-forward
network, each wrapped with layer normalization and a residual connection.

This implementation uses the **Pre-Norm** variant (also called Pre-LN),
where layer normalization is applied *before* each sub-layer rather than
after. Pre-Norm is preferred in modern Transformers (GPT-2, GPT-3, LLaMA,
etc.) because it stabilizes training without warm-up schedules.

Pre-Norm Transformer block:
    norm_x     = LayerNorm(x)
    x          = x + Attention(norm_x)       # residual after attention
    norm_x     = LayerNorm(x)
    x          = x + FFN(norm_x)             # residual after FFN
    return x

Contrast with the original Post-Norm (Vaswani et al., 2017):
    x = LayerNorm(x + Attention(x))
    x = LayerNorm(x + FFN(x))

The block is the building block that gets stacked N times in Chapter 8
to form a complete Transformer model.

Builds on:
  - 04_multi_head_attention.py  (MultiHeadAttention)
  - 05_feed_forward.py          (FeedForward)
  - 06_layer_norm_residual.py   (LayerNorm, residual_add)
"""

import importlib
import sys
import os

# ---------------------------------------------------------------------------
# Import components from previous chapters
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(__file__))

mha_mod = importlib.import_module('04_multi_head_attention')
MultiHeadAttention = mha_mod.MultiHeadAttention

ffn_mod = importlib.import_module('05_feed_forward')
FeedForward = ffn_mod.FeedForward

ln_mod = importlib.import_module('06_layer_norm_residual')
LayerNorm    = ln_mod.LayerNorm
residual_add = ln_mod.residual_add


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock:
    """One complete Transformer layer: Pre-Norm attention + Pre-Norm FFN.

    Each sub-layer (attention, FFN) is preceded by a LayerNorm and followed
    by a residual addition, forming the standard "Pre-LN" building block.

    Parameters
    ----------
    embed_dim : int  — token embedding dimensionality (d_model)
    num_heads : int  — number of parallel attention heads
    ff_dim    : int or None — inner dim of the FFN (default: 4 * embed_dim)
    seed      : int  — base random seed; attention gets seed, FFN gets seed+1
    """

    def __init__(self, embed_dim, num_heads, ff_dim=None, seed=42):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Sub-layer 1: Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, seed)

        # Sub-layer 2: Position-wise feed-forward network
        self.ffn = FeedForward(embed_dim, ff_dim, seed + 1)

        # Pre-norm layers (one per sub-layer)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """Run a single Transformer block over an input sequence.

        Pre-Norm pattern:
            1.  norm_x     = norm1(x)                    — normalize first
            2.  attn_out   = attention(norm_x, mask)     — attend
            3.  x          = x + attn_out                — residual
            4.  norm_x     = norm2(x)                    — normalize again
            5.  ffn_out    = ffn(norm_x)                 — transform
            6.  x          = x + ffn_out                 — residual
            7.  return x

        Parameters
        ----------
        x    : list[list[float]] — shape [seq_len x embed_dim]
        mask : list[list[bool]] or None — optional causal mask

        Returns
        -------
        list[list[float]] — shape [seq_len x embed_dim]
        """
        # --- Sub-layer 1: Self-attention with Pre-Norm ---
        norm_x     = self.norm1.forward_sequence(x)
        attn_out   = self.attention.forward(norm_x, mask)
        x          = residual_add(x, attn_out)

        # --- Sub-layer 2: FFN with Pre-Norm ---
        norm_x     = self.norm2.forward_sequence(x)
        ffn_out    = self.ffn.forward(norm_x)
        x          = residual_add(x, ffn_out)

        return x


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 "phase1-from-scratch/level-c-reader/07_transformer_block.py"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("Chapter 7 — Transformer Block Demo")
    print("Pre-Norm attention + FFN from scratch")
    print("=" * 60)

    embed_dim = 16
    num_heads = 4
    seq_len   = 5

    block = TransformerBlock(embed_dim, num_heads, seed=42)
    print(f"\nConfiguration:")
    print(f"  embed_dim : {embed_dim}")
    print(f"  num_heads : {num_heads}")
    print(f"  head_dim  : {embed_dim // num_heads}")
    print(f"  ff_dim    : {block.ffn.ff_dim}  (= 4 × embed_dim)")

    # Build sample input
    rng = random.Random(7)
    x = [[rng.gauss(0, 0.5) for _ in range(embed_dim)] for _ in range(seq_len)]

    print(f"\nInput shape  : {len(x)} × {len(x[0])}")
    out = block.forward(x)
    print(f"Output shape : {len(out)} × {len(out[0])}")

    # Verify shape preserved
    assert len(out) == seq_len and len(out[0]) == embed_dim, "Shape mismatch!"
    print("\nOutput (first 4 dims of each position):")
    for i, vec in enumerate(out):
        snippet = ", ".join(f"{v:.4f}" for v in vec[:4])
        print(f"  pos {i}: [{snippet}, ...]")

    # Show with causal mask
    attn_mod = importlib.import_module('03_self_attention')
    mask = attn_mod.create_causal_mask(seq_len)
    out_masked = block.forward(x, mask)
    print(f"\nWith causal mask — output shape: {len(out_masked)} × {len(out_masked[0])}")

    # Determinism check
    out2 = block.forward(x)
    same = all(
        abs(out[i][j] - out2[i][j]) < 1e-12
        for i in range(seq_len)
        for j in range(embed_dim)
    )
    print(f"\nDeterministic (same input → same output): {same}")

    print("\n" + "=" * 60)
    print("Transformer block complete. Chapter 8: Stack N blocks → full model.")
    print("=" * 60)
