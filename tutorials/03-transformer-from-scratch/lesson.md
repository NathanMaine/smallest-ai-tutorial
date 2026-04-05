# Lesson: Transformer from Scratch

The detailed walkthrough. Read this alongside the code.

---

## Chapter 1: Embeddings

The first thing a Transformer does is convert tokens (letters, words, whatever) into dense vectors. This is the **embedding layer**.

A `Vocabulary` maps tokens to integer indices. An `EmbeddingLayer` maps those integers to vectors of size `embed_dim`. The vectors start random and are learned during training.

Why vectors instead of one-hot encodings? One-hot vectors are sparse (one 1 in 26 for letters). Learned embeddings are dense (every dimension matters) and can represent semantic similarity — similar tokens get similar vectors.

---

## Chapter 2: Positional Encoding

The Transformer processes all positions in parallel. Unlike the LSTM which reads left-to-right and gets position for free, the Transformer has no inherent notion of order.

Sinusoidal positional encodings inject position information:
```
PE[pos][2i]   = sin(pos / 10000^(2i/d_model))
PE[pos][2i+1] = cos(pos / 10000^(2i/d_model))
```

These are *added* to the token embeddings before the Transformer layers.

Why sinusoids? They have a property that makes relative position arithmetic possible: `PE[pos+k]` can be expressed as a linear function of `PE[pos]`. Modern models often use learned positional embeddings instead, but sinusoidal encodings are parameter-free and generalize to lengths longer than training sequences.

---

## Chapter 3: Self-Attention

The core innovation. Every position can directly attend to every other position.

Each input vector gets projected into three spaces:
- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"  
- **Value**: "What information do I share?"

The attention score between positions i and j:
```
score(i, j) = dot(Q[i], K[j]) / sqrt(d_k)
```

After softmax, scores become weights summing to 1. The output is a weighted blend of Value vectors:
```
output[i] = sum(attention_weight[i][j] * V[j] for all j)
```

### The Scaling Factor

Why divide by `sqrt(d_k)`? For large d_k, the dot products grow in magnitude, which pushes softmax into a region where gradients are tiny (the softmax output becomes near-one-hot). Dividing by `sqrt(d_k)` keeps scores well-scaled.

### Causal Masking

For language models, position i should not see positions j > i (future tokens). The causal mask sets those scores to -1e9 before softmax, making their attention weights effectively zero.

---

## Chapter 4: Multi-Head Attention

Run attention multiple times in parallel with different Q/K/V projections. Each "head" can attend to different aspects of the input:

```
head_i = Attention(Q·W_q_i, K·W_k_i, V·W_v_i)
MHA = concat(head_1, ..., head_h) · W_out
```

Why multiple heads? A single attention head computes one way of blending values. Multiple heads let the model simultaneously attend to syntactic structure, semantic similarity, and positional proximity (for example) — different aspects in parallel.

---

## Chapters 5–6: FFN and LayerNorm

**FeedForward Network**: A two-layer MLP applied *independently* to each position:
```
FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
```
The inner dimension is typically 4x the embed_dim. This is where most of the "computation" in a Transformer happens.

**Layer Normalization**: Normalizes each vector to have mean ≈ 0 and variance ≈ 1, then scales and shifts with learned parameters. Applied before each sub-layer in the Pre-Norm variant.

**Residual connections**: `x = x + sublayer(x)`. Gradients flow through the identity shortcut, bypassing the sub-layer. Essential for deep networks.

---

## Chapter 7: Transformer Block

One complete block:
```python
norm_x   = norm1(x)            # normalize
x        = x + attention(norm_x)  # attend + residual
norm_x   = norm2(x)            # normalize again
x        = x + ffn(norm_x)       # transform + residual
```

Stack N of these for a full Transformer (Chapter 8).

---

## Chapter 9: Language Model Training

A language model predicts the next token given the current sequence. We train on letter sequences using the standard approach:
- Input:  `[a, b, c, d]`
- Target: `[b, c, d, <eos>]`

At each position, predict the next token. This is **next-token prediction**, the training objective behind GPT-style models.

Our Transformer is small (2 layers, 4 heads, embed_dim=16) but the training loop is identical to what runs in much larger models. The difference is scale, not structure.
