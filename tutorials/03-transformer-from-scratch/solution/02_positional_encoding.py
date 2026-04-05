"""
Chapter 2 — Positional Encoding
=================================

What this module teaches
-------------------------
Transformers process *all* tokens in parallel — unlike RNNs, which see tokens
one at a time. This is powerful for parallelism but creates a problem: the model
has no built-in notion of *order*. Two sentences with the same words in a
different order would produce identical representations.

**Positional encoding** solves this by adding a unique signal to each token's
embedding that encodes its position in the sequence.

The sinusoidal scheme (Vaswani et al., "Attention Is All You Need", 2017):

    PE(pos, 2i)   = sin(pos / 10000^(2i / embed_dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / embed_dim))

Where:
  - pos     = position index (0-based row of the sequence)
  - i       = dimension pair index (0 ≤ i < embed_dim // 2)
  - embed_dim = total embedding dimensionality

Properties of this encoding:
  1. Every position gets a *unique* vector — no two positions are identical.
  2. The encoding is *deterministic* — no learned parameters.
  3. sin/cos values stay in [-1, 1], keeping the scale well-matched to Xavier-
     initialised embeddings.
  4. The model can *extrapolate* to longer sequences at inference time because
     the formula generalises beyond max_seq_len.

Usage:
    pe = sinusoidal_encoding(max_seq_len=50, embed_dim=16)
    # pe[pos] is the position vector for token at position pos

    enriched = add_position_info(embeddings, pe)
    # enriched[t] = embeddings[t] + pe[t]
"""

import math


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_encoding(max_seq_len, embed_dim):
    """Build a sinusoidal positional encoding matrix.

    Returns a matrix of shape [max_seq_len x embed_dim] where row *pos* is
    the position encoding for sequence position *pos*.

    The formula for each element:
        PE[pos, 2i]   = sin(pos / 10000^(2i / embed_dim))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / embed_dim))

    Args:
        max_seq_len: int — number of positions to pre-compute (rows)
        embed_dim:   int — dimensionality of each position vector (cols)

    Returns:
        list[list[float]] — shape [max_seq_len][embed_dim]
    """
    pe = []

    for pos in range(max_seq_len):
        row = [0.0] * embed_dim

        for i in range(embed_dim // 2):
            # Denominator: 10000^(2i / embed_dim)
            denominator = math.pow(10000.0, (2 * i) / embed_dim)
            angle = pos / denominator

            row[2 * i]     = math.sin(angle)   # even dimension
            row[2 * i + 1] = math.cos(angle)   # odd dimension

        # If embed_dim is odd, the last element stays 0.0 (sin of 0 for the
        # extra unpaired dimension — a graceful fallback).

        pe.append(row)

    return pe


# ---------------------------------------------------------------------------
# Combining embeddings with positional encodings
# ---------------------------------------------------------------------------

def add_position_info(embeddings, pe_matrix):
    """Element-wise addition of token embeddings and their position encodings.

    For each position t:
        result[t][d] = embeddings[t][d] + pe_matrix[t][d]

    This injects positional information directly into the embedding space.
    The Transformer's attention mechanism then sees both *what* each token is
    (embedding) and *where* it appears (position encoding).

    Args:
        embeddings: list[list[float]] — token embedding vectors, shape [seq_len x D]
        pe_matrix:  list[list[float]] — positional encoding matrix, shape [≥seq_len x D]

    Returns:
        list[list[float]] — new list of same shape as embeddings
    """
    result = []
    for t, emb in enumerate(embeddings):
        pe_row = pe_matrix[t]
        combined = [emb[d] + pe_row[d] for d in range(len(emb))]
        result.append(combined)
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MAX_SEQ_LEN = 10
    EMBED_DIM   = 8

    print("=" * 60)
    print("Chapter 2: Positional Encoding Demo")
    print("=" * 60)

    pe = sinusoidal_encoding(MAX_SEQ_LEN, EMBED_DIM)

    print(f"\nSinusoidal PE matrix [{MAX_SEQ_LEN} positions x {EMBED_DIM} dims]:")
    print(f"{'pos':>4s}  " + "  ".join(f"d{d:<4d}" for d in range(EMBED_DIM)))
    print("-" * (6 + 8 * EMBED_DIM))
    for pos in range(MAX_SEQ_LEN):
        vals = "  ".join(f"{v:+.3f}" for v in pe[pos])
        print(f"{pos:>4d}  {vals}")

    # Demonstrate add_position_info with a trivial zero embedding
    print(f"\nadd_position_info example (zero embeddings + PE = PE itself):")
    zero_embeddings = [[0.0] * EMBED_DIM for _ in range(4)]
    enriched = add_position_info(zero_embeddings, pe)
    for t, vec in enumerate(enriched):
        formatted = [f"{v:+.4f}" for v in vec]
        print(f"  pos={t}: [{', '.join(formatted)}]")

    # Highlight the difference between two positions
    print(f"\nPosition 0 vs Position 1 (first 4 dims):")
    for pos in [0, 1]:
        snippet = [f"{v:+.4f}" for v in pe[pos][:4]]
        print(f"  PE[{pos}][:4] = [{', '.join(snippet)}]")
