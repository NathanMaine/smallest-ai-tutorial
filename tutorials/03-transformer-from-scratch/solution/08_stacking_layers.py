"""
Chapter 8 — Stacking Layers: The Full Transformer
===================================================

What this module teaches
-------------------------
We now have every individual component. This chapter assembles them into a
complete, end-to-end Transformer model:

    token_indices
        ↓
    EmbeddingLayer          — map token IDs to dense vectors
        ↓
    + Positional Encoding   — inject sequence-order information
        ↓
    TransformerBlock × N    — N stacked Pre-Norm attention + FFN layers
        ↓
    LayerNorm (final)       — stabilise the top of the stack
        ↓
    Output projection       — linear map from embed_dim → vocab_size logits
        ↓
    logits [seq_len × vocab_size]

The logits represent un-normalised scores over the vocabulary at every
position. A softmax converts them to probabilities; argmax picks the most
likely next token. This is the core inference loop of every GPT-style LM.

Architecture details
---------------------
  - Embedding dim (d_model):  configurable, default 64
  - Attention heads:          configurable, default 4
  - Stacked blocks (layers):  configurable, default 2
  - FFN inner dim:            4 × embed_dim (default)
  - Max sequence length:      128 (pre-computed positional encodings)
  - Output projection:        W_out [vocab_size × embed_dim] + b_out [vocab_size]

Builds on:
  - level-a-abcs/01_math_foundations.py  (dot_product)
  - level-c-reader/01_embeddings.py      (Vocabulary, EmbeddingLayer)
  - level-c-reader/02_positional_encoding.py  (sinusoidal_encoding, add_position_info)
  - level-c-reader/03_self_attention.py  (create_causal_mask)
  - level-c-reader/06_layer_norm_residual.py  (LayerNorm)
  - level-c-reader/07_transformer_block.py    (TransformerBlock)
"""

import importlib
import sys
import os
import random

# ---------------------------------------------------------------------------
# Import all components
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(__file__))

math_fn    = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product

embed_mod      = importlib.import_module('01_embeddings')
Vocabulary     = embed_mod.Vocabulary
EmbeddingLayer = embed_mod.EmbeddingLayer

pe_mod           = importlib.import_module('02_positional_encoding')
sinusoidal_encoding = pe_mod.sinusoidal_encoding
add_position_info   = pe_mod.add_position_info

block_mod        = importlib.import_module('07_transformer_block')
TransformerBlock = block_mod.TransformerBlock

ln_mod    = importlib.import_module('06_layer_norm_residual')
LayerNorm = ln_mod.LayerNorm

attn_mod          = importlib.import_module('03_self_attention')
create_causal_mask = attn_mod.create_causal_mask


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class Transformer:
    """A complete autoregressive Transformer language model.

    Processes a sequence of token indices and returns logits over the
    vocabulary at every position (i.e., predicted scores for the next token).

    Parameters
    ----------
    vocab_size  : int  — number of tokens in the vocabulary
    embed_dim   : int  — token + positional embedding dimensionality (d_model)
    num_heads   : int  — attention heads per block
    num_layers  : int  — number of stacked TransformerBlocks
    ff_dim      : int or None — FFN inner dim (default: 4 * embed_dim)
    max_seq_len : int  — maximum sequence length for positional encoding
    seed        : int  — base random seed; each sub-component gets seed+offset
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=None,
        max_seq_len=128,
        seed=42,
    ):
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.num_layers  = num_layers
        self.max_seq_len = max_seq_len

        # 1. Token embedding table [vocab_size × embed_dim]
        self.embedding = EmbeddingLayer(vocab_size, embed_dim, seed)

        # 2. Pre-computed positional encoding [max_seq_len × embed_dim]
        self.pe = sinusoidal_encoding(max_seq_len, embed_dim)

        # 3. Stack of N Transformer blocks (each block gets a unique seed)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, seed + i)
            for i in range(num_layers)
        ]

        # 4. Final layer norm (applied after all blocks)
        self.final_norm = LayerNorm(embed_dim)

        # 5. Output projection: maps each hidden state to vocab_size logits
        #    W_out : [vocab_size × embed_dim]
        #    b_out : [vocab_size]
        rng   = random.Random(seed + num_layers + 1)
        import math
        limit = math.sqrt(6.0 / (embed_dim + vocab_size))
        self.W_out = [
            [rng.uniform(-limit, limit) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]
        self.b_out = [0.0] * vocab_size

    def forward(self, token_indices):
        """Run a full forward pass over a token sequence.

        Steps:
          1. Embed token indices → [seq_len × embed_dim]
          2. Add positional encoding
          3. Apply causal mask through all N blocks
          4. Final LayerNorm
          5. Project each position to vocabulary logits

        Parameters
        ----------
        token_indices : list[int] — token IDs (e.g. from Vocabulary.encode_sentence)

        Returns
        -------
        list[list[float]] — shape [seq_len × vocab_size], unnormalised logits
        """
        seq_len = len(token_indices)

        # Step 1 + 2: Embeddings + positional encoding
        embeddings = self.embedding.forward(token_indices)
        x = add_position_info(embeddings, self.pe[:seq_len])

        # Step 3: Pass through all Transformer blocks with causal mask
        mask = create_causal_mask(seq_len)
        for block in self.blocks:
            x = block.forward(x, mask)

        # Step 4: Final layer norm
        x = self.final_norm.forward_sequence(x)

        # Step 5: Project to vocabulary — one logit vector per position
        logits = []
        for i in range(seq_len):
            pos_logits = [
                dot_product(self.W_out[v], x[i]) + self.b_out[v]
                for v in range(self.vocab_size)
            ]
            logits.append(pos_logits)

        return logits

    def get_params_count(self):
        """Count the total number of scalar parameters in the model.

        Returns
        -------
        int — total parameter count (embedding table + all blocks + output projection)
        """
        count = 0

        # Token embeddings
        count += self.vocab_size * self.embed_dim

        # Each Transformer block
        for block in self.blocks:
            d = self.embed_dim
            h = block.ffn.ff_dim

            # MultiHeadAttention: each head has W_q, W_k, W_v each [head_dim × embed_dim]
            # plus W_output [embed_dim × embed_dim]
            num_heads = block.attention.num_heads
            head_dim  = block.attention.head_dim
            # Per head: 3 weight matrices [head_dim × embed_dim]
            count += num_heads * 3 * (head_dim * d)
            # Output projection [embed_dim × embed_dim]
            count += d * d

            # FeedForward: W1 [ff_dim × embed_dim] + b1 [ff_dim]
            #               W2 [embed_dim × ff_dim] + b2 [embed_dim]
            count += h * d + h        # W1 + b1
            count += d * h + d        # W2 + b2

            # Two LayerNorms: gamma + beta, each [embed_dim]
            count += 2 * (d + d)

        # Final LayerNorm
        count += 2 * self.embed_dim

        # Output projection W_out [vocab_size × embed_dim] + b_out [vocab_size]
        count += self.vocab_size * self.embed_dim + self.vocab_size

        return count


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 "phase1-from-scratch/level-c-reader/08_stacking_layers.py"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 8 — Full Stacked Transformer Demo")
    print("Embeddings → PE → N Blocks → Logits, pure Python")
    print("=" * 60)

    # Build a tiny vocabulary
    vocab = Vocabulary()
    vocab.build([
        "the cat sat on the mat",
        "the dog ran in the park",
        "transformers use attention to read sentences",
        "stacking layers makes a full language model",
    ])
    vocab_size = vocab.size
    print(f"\nVocabulary size : {vocab_size}")

    # Create transformer
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=64,
        seed=42,
    )
    total_params = model.get_params_count()
    print(f"Model config    : embed_dim=32, heads=4, layers=2")
    print(f"Total parameters: {total_params:,}")

    # Encode a sample sentence and run forward pass
    sentence = "the cat sat on the mat"
    token_ids = vocab.encode_sentence(sentence)
    print(f"\nInput sentence  : '{sentence}'")
    print(f"Token IDs       : {token_ids}")
    print(f"Sequence length : {len(token_ids)}")

    logits = model.forward(token_ids)
    print(f"\nOutput shape    : {len(logits)} × {len(logits[0])}")
    print(f"  (seq_len={len(logits)}, vocab_size={len(logits[0])})")

    print("\nLogits at each position (first 5 vocab scores):")
    for i, pos_logits in enumerate(logits):
        preview = ", ".join(f"{v:.4f}" for v in pos_logits[:5])
        print(f"  pos {i}: [{preview}, ...]")

    # Show predicted next-token at each position
    print("\nArgmax predictions (most likely next token at each position):")
    for i, pos_logits in enumerate(logits):
        best_idx  = max(range(vocab_size), key=lambda v: pos_logits[v])
        best_word = vocab.decode(best_idx)
        print(f"  pos {i}: token {best_idx} → '{best_word}'")

    # Different input → different output
    sentence2 = "transformers use attention"
    ids2 = vocab.encode_sentence(sentence2)
    logits2 = model.forward(ids2)
    print(f"\nSecond input '{sentence2}'")
    print(f"Output shape: {len(logits2)} × {len(logits2[0])}")

    print("\n" + "=" * 60)
    print("Full Transformer assembled. Architecture complete.")
    print("=" * 60)
