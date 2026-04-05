# Tutorial 03: Transformer from Scratch

**Build the architecture behind GPT, BERT, and LLaMA — from pure Python.**

---

## What Makes This Tricky

The Transformer has more moving parts than the MLP or LSTM. You're not just wiring layers together — you're wiring them together with attention, and attention has several subtle details that make it hard to get right on the first try.

The one that tripped me up: scaling. The attention scores are divided by `sqrt(d_k)` before the softmax. This seems like a small detail, but without it, the dot products grow with `d_k`, which pushes the softmax into a regime where gradients vanish. It's called "scaled dot-product attention" and the scaling is non-negotiable.

The second tricky thing: positional encoding. The Transformer processes all positions in parallel (unlike the LSTM which reads them in sequence), so it has no inherent notion of order. Positional encodings inject position information as sinusoidal patterns added to the input embeddings. Understanding *why* sinusoids are chosen (they have unique properties for relative position arithmetic) took me a while.

The third: causal masking. When you want the model to predict the next token, it must not be allowed to "look ahead" at future tokens during training. The causal mask sets future positions to `-infinity` before softmax so they receive zero attention weight.

---

## What Surprised Me

How much less code the Transformer is than I expected. After seeing it described as a breakthrough architecture, I assumed it would be hundreds of lines. The core of it — self-attention, the transformer block — is actually quite compact. The complexity is conceptual, not volumetric.

Also: the Pre-Norm vs Post-Norm choice. The original paper (Vaswani et al., 2017) used Post-Norm (normalize *after* the attention). Modern models (GPT-2, GPT-3, LLaMA) use Pre-Norm (normalize *before*). Pre-Norm is more stable and doesn't need warm-up schedules. This repo implements Pre-Norm.

---

## What You'll Build

A complete Transformer that:
- Embeds tokens into learned vector representations
- Adds sinusoidal positional encodings to preserve order information
- Applies multi-head self-attention with causal masking
- Processes through a position-wise feed-forward network
- Uses layer normalization and residual connections (Pre-Norm variant)
- Can be trained as a language model on letter sequences

Nine chapters, building attention piece by piece.

---

## Chapter Overview

| Chapter | File | What It Covers |
|---------|------|----------------|
| 1 | `01_embeddings.py` | Vocabulary, token-to-vector lookup table |
| 2 | `02_positional_encoding.py` | Sinusoidal positional encodings |
| 3 | `03_self_attention.py` | Scaled dot-product attention + causal mask |
| 4 | `04_multi_head_attention.py` | Parallel attention heads + output projection |
| 5 | `05_feed_forward.py` | Position-wise FFN (two linear layers, ReLU) |
| 6 | `06_layer_norm_residual.py` | LayerNorm and residual connections |
| 7 | `07_transformer_block.py` | One complete block: Pre-Norm + attention + FFN |
| 8 | `08_stacking_layers.py` | N stacked blocks = full Transformer |
| 9 | `09_final_project.py` | Language model training on letter sequences |

---

## The Core Idea in One Paragraph

The key insight of the Transformer is this: instead of processing tokens one at a time (like an LSTM), process them *all at once* and let each token directly query every other token. The Query, Key, Value mechanism is a differentiable lookup: each position asks "what am I looking for?" (Query), each position advertises "what do I contain?" (Key), and the dot product of Q and K tells us how much attention to pay. The Values are then blended proportionally. Multi-head attention runs this in parallel for several different Q/K/V projections, letting the model attend to different aspects simultaneously.

---

## Running the Complete Solution

```bash
python3 tutorials/03-transformer-from-scratch/solution/09_final_project.py
```

This trains a small Transformer as a language model on letter sequences.

---

## What's Next

Tutorial 04 puts all four architectures side by side and compares them: MoE, Mamba, BitNet, and quantized Transformer. It's less "follow these steps" and more "here's the benchmark suite — explore."

→ [Tutorial 04: Comparison Study](../04-comparison-study/README.md)
