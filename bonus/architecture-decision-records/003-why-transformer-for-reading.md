# ADR 003: Why Transformer for Level C (Reading)

**Date:** 2026-04-04  
**Status:** Accepted  
**Deciders:** Nathan Maine  

## Context

Level C introduces sentence and story understanding—a fundamentally harder problem than letter classification (Level A) or phonics (Level B). A reader must:

1. **Understand long-range dependencies:** Connecting subject to verb across many words (e.g., "The big dog that ran in the park yesterday is sleeping")
2. **Handle variable-length sequences:** Sentences and stories have different lengths
3. **Maintain positional awareness:** Word order matters (e.g., "dog bites man" vs. "man bites dog")
4. **Attend to relevant context:** Not all words are equally important for understanding (e.g., articles vs. nouns)

Previous architectures have limitations:

- **RNNs (Level B):** Sequential processing is slow; gradients vanish over long sequences; difficult to parallelize
- **Larger MLPs:** Fully-connected layers have no sequential structure; they'd need massive parameter counts for long sequences
- **CNNs:** Limited receptive field without many layers; not natural for variable-length text

We need an architecture that handles **long-range dependencies efficiently while maintaining interpretability**.

## Decision

**Use Transformers (encoder-only) as the core architecture for Level C.**

## Rationale

### 1. Solves Long-Range Dependencies

Transformers use **self-attention**, which allows every token to directly attend to every other token in a single layer. This solves the vanishing gradient problem that plagues RNNs:

- **RNN (2-layer):** After 10 steps, gradient has multiplied through ~20 Jacobians. If each ≤0.9, gradient ≈ 0.9^20 ≈ 0.12 (severe attenuation).
- **Transformer (2-layer):** Every token can attend directly to earlier tokens. Gradients flow in one "hop" via attention weights.

For sentence understanding (typical length 10-30 words), this is a game-changer.

### 2. Parallel Computation

RNNs process sequentially: token 1 → token 2 → token 3 → ... Each step waits for the previous. Transformers process all tokens at once:

- **RNN training time:** O(seq_len) (sequential)
- **Transformer training time:** O(1) in sequence depth (all tokens in parallel, but O(seq_len^2) in memory for attention)

For our small sequences (32 tokens), this is fast.

### 3. Interpretability

Attention weights directly show what the model is attending to:

```
Token: "dog"
Attention weight to "big":  0.7
Attention weight to "ran":  0.6
Attention weight to "park": 0.2
Attention weight to "the":  0.05
```

This makes debugging and understanding the model's decisions much easier than RNN hidden states, which are opaque.

### 4. Natural for Text

Transformers are **the standard for NLP** because they align with how humans read:

- We don't read left-to-right sequentially; we skip, scan, and refer back.
- Attention models this behavior directly.

### 5. Scalable Foundation

Modern LLMs (GPT, BERT, Claude, LLaMA) are all transformer-based. Learning transformers from scratch prepares you for understanding and building production systems.

## Alternatives Considered

### Alternative 1: Bigger RNN (GRU/LSTM with more layers)

**Pros:**
- Simpler to understand (just recurrence + gating)
- Fewer hyperparameters

**Cons:**
- Still struggles with long-range dependencies (vanishing gradients don't fully disappear with gating)
- Sequential processing is slow
- No built-in interpretability
- Not the modern standard

**Verdict:** REJECTED. RNNs hit their architectural limits for this task.

### Alternative 2: Larger MLP

**Pros:**
- Simpler than transformers
- Trains fast (fully dense)

**Cons:**
- No sequential structure; would need O(seq_len^2) parameters to connect all position pairs
- Not natural for variable-length sequences
- Completely opaque (no interpretability)
- Would overfit immediately on small dataset

**Verdict:** REJECTED. Fundamentally wrong for sequences.

### Alternative 3: CNN with Large Receptive Field

**Pros:**
- Can be trained in parallel
- Efficient memory usage

**Cons:**
- Still requires stacking many layers to reach long-range dependencies (receptive field grows linearly per layer)
- Not natural for text (spatial priors don't apply)
- Less interpretable than attention

**Verdict:** REJECTED. Attention is more direct.

## Consequences

### Positive

1. **Student learns the modern standard:** Transformers are everywhere in production ML.
2. **Clear path to LLMs:** Level D can add decoding (GPT-style), Phase 2 can add pre-training.
3. **Interpretability:** Attention weights can be visualized and understood.
4. **Efficient for our scale:** 32-token sequences, 128-dim embeddings, 2-3 blocks is fast.

### Negative

1. **Most complex so far:** 11 chapters vs. 8 (Level A) and 8 (Level B). Requires careful explanation.
2. **More hyperparameters:** Num heads, head dimension, feed-forward hidden size, etc. More tuning needed.
3. **O(seq_len^2) attention:** For very long sequences (>1024), attention becomes a bottleneck. We'll note this for Phase 2.
4. **Requires positional encoding:** RNNs "remember" position implicitly via recurrence; transformers need explicit position signals.

## Validation

We will validate this decision by:

1. **Implementing chapter-by-chapter:** Each chapter should run and show clear progress (loss decreasing, attention becoming meaningful).
2. **Attention visualization:** Plot attention weights; verify they focus on semantically relevant tokens (e.g., attending to adjectives when processing nouns).
3. **Baseline comparison (Phase 2):** Compare a transformed-trained model vs. an RNN baseline on the same data. Transformer should be faster and reach higher accuracy.
4. **Student feedback:** Verify that students find transformers understandable and the 11 chapters digestible.

## Related Decisions

- **ADR 001 (MLPs for ABCs):** MLPs lack structure; transformers add the right structure (attention) for sequences.
- **ADR 002 (RNNs for Phonics):** RNNs are good for character sequences; transformers are better for word sequences and longer dependencies.
- **Future (Decoders):** Once transformers are mastered, adding decoder stacks (Level D) is straightforward.

## References

- **Vaswani et al. (2017).** "Attention Is All You Need." NIPS. The original transformer paper.
- **Devlin et al. (2018).** "BERT: Pre-training of Deep Bidirectional Transformers." NIPS. Encoder-only transformers for understanding.
- **Radford et al. (2019).** "Language Models are Unsupervised Multitask Learners." OpenAI. GPT-2. Decoder-only for generation.
- **Raganato & Tiedemann (2018).** "Analyzing the Source and Target Contributions to Predictions in Neural Machine Translation." ACL. Attention visualization.
