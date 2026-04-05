# ADR-002: Why RNN/LSTM for Level B (Phonics)

**Status:** Accepted  
**Date:** 2026-04-04  
**Context:** Level B teaches letter-to-phoneme mapping on sequences (e.g., "cat" → /k/-/æ/-/t/). Words have variable length (3-8 letters). Phonetic rules (digraphs, vowel teams, silent-e) are context-dependent.

## Decision

Use a **Long Short-Term Memory (LSTM) Recurrent Neural Network** for Level B.

## Rationale

### 1. Sequences Require Memory

**Level A was classification**: one fixed-size input (26-dim one-hot letter) → one output class. MLPs are ideal.

**Level B is sequence-to-sequence**: variable-length letter sequences → variable-length phoneme sequences. Example:
- Input: c, a, t (3 letters)
- Output: /k/, /æ/, /t/ (3 phonemes)
- Problem: The phoneme for "a" depends on neighboring "c" and "t". An MLP discards order.

**RNN solution**: Hidden state acts as memory. At each time step, the network updates a hidden vector summarizing "what letters have we seen so far?" This allows:
- Letter 1 ("c") influences letter 2 ("a") influences letter 3 ("t")
- Digraphs: "sh" is not /s/ + /h/, but a single unit /ʃ/
- Context-dependent rules: Final consonants modify vowel pronunciation

### 2. Phonetic Rules Are Context-Dependent

English phonetics are sequential and context-sensitive:

- **Digraphs** (sh, ch, th, wh, ph, ck, ng): Two letters → one phoneme. "ship" ≠ "s" + "h". An MLP has no way to group ("s", "h") as a unit.
- **Vowel teams** (ai, ea, oa, ee): Same as digraphs, but for vowels. "rain" has /eɪ/ for "ai", not separate vowels.
- **Silent-e rule**: "hope" vs. "hop". The final "e" modifies the vowel's pronunciation. This is a backward dependency—RNNs can model it.
- **Consonant clusters** (bl, br, dr, fr, gr, st, tr): Two consonants blend. "blue" ≠ separate /b/ + /l/.

RNNs learn these patterns implicitly. Feed training data with digraphs; LSTM cells discover that "s" followed by "h" predicts a different phoneme distribution.

### 3. Variable-Length Sequences

Words are not all 3 letters:
- "bat" (3)
- "blend" (5)
- "string" (6)

MLPs require fixed input size. You'd need to **pad** all sequences to 8 letters and add masking logic. RNNs handle variable length naturally—just process until end-of-sequence.

### 4. Why LSTM, Not Vanilla RNN?

Vanilla RNNs have a critical flaw: **vanishing gradients**. When you backprop through many time steps, gradients shrink exponentially. Weights stop learning.

LSTM gates solve this:
- **Forget gate** (f): Decide what to discard from cell state
- **Input gate** (i): Decide what new information to store
- **Output gate** (o): Decide what to emit from cell state
- **Cell state** (c): Highway of information, modified additively (resistant to vanishing)

This lets LSTM learn long-range dependencies (e.g., a vowel depends on a consonant 2 steps away). Vanilla RNN cannot.

For our phonics task, words are short (3-8 letters), so vanilla RNN *might* work. But LSTM is the standard and teaches the right lessons for scaling to Level C and D.

## Alternatives Considered

### MLP (Fully Connected)
- **Pros**: Simplest to implement. Reuses Layer abstraction from Level A.
- **Cons**: 
  - Requires fixed input size. Must pad all words to 8 letters → wasteful.
  - No notion of sequence. Cannot learn digraphs implicitly.
  - Cannot model left-to-right blending naturally.
- **Verdict**: Insufficient for the problem.

### CNN (Convolutional)
- **Pros**: Handles variable-length sequences via global pooling. Can learn local patterns (digraphs as 2-filters).
- **Cons**:
  - Designed for spatial locality (images), not temporal dependencies.
  - Convolution is not naturally left-to-right; it's symmetric (reads both sides simultaneously).
  - Cannot easily model backward dependencies (silent-e) without bidirectional padding.
  - Overkill for short sequences.
- **Verdict**: Applicable, but worse fit than RNN.

### Transformer (Attention)
- **Pros**: Parallel processing. Can model long-range dependencies with attention.
- **Cons**:
  - Extreme overkill for 3-8 letter words. Attention overhead (O(n²)) is wasted.
  - Requires positional encoding; adds unnecessary complexity.
  - Harder to implement from scratch.
  - Hides sequential nature of phonics (token independence is not realistic).
- **Verdict**: Premature optimization. Level D unifies multi-task learning; Transformer is better there.

### Lookup Table
- **Pros**: Trivial to implement. Perfect accuracy on training data.
- **Cons**:
  - Teaches nothing about neural networks.
  - Zero generalization to unseen words.
  - Not the goal of an educational project.
- **Verdict**: Defeated by project philosophy.

## Consequences

### Benefits
1. **Educational**: Teaches recurrence, gates, and BPTT—skills needed for language models, speech, time series.
2. **Generalizable**: Students learn LSTM fundamentals reusable in Level C (word classification) and D (multi-task).
3. **Right-sized**: ~100k parameters fits on a laptop. Train in <1 minute on CPU.
4. **Empirical win**: RNNs implicitly learn digraphs without explicit rules. Network discovers "sh" → /ʃ/ from data alone.
5. **Phonetically sound**: Mirrors linguistic theories (autosegmental phonology, feature geometry). Sequence structure = linguistic structure.

### Drawbacks
1. **Gradient complexity**: BPTT is harder to understand than backprop. Truncation, clipping, and masks add bookkeeping.
2. **Slower inference**: Recurrence is not parallelizable. Inference is O(n) rather than O(1) like MLPs.
3. **Vanishing/exploding gradients**: Must implement gradient clipping and weight initialization carefully.
4. **Teacher forcing**: Seq2seq training requires different inference procedure. Distribution shift risk.

### Implementation Obligations
- Implement LSTM cells from scratch (forward + backward)
- Implement Backpropagation Through Time (BPTT)
- Implement gradient clipping and layer normalization
- Handle variable-length sequences with masking
- Implement teacher forcing for training
- Analyze LSTM gates to extract learned rules

## References

- **LSTM paper**: Hochreiter & Schmidhuber (1997). "Long Short-Term Memory". Neural Computation.
- **Understanding LSTMs**: Colah's blog (2015). https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Seq2Seq**: Sutskever, Vanhoucke, Le (2014). "Sequence to Sequence Learning with Neural Networks". NIPS.
- **Phonological theory**: Anderson, John M. (2008). "The Structure and Loss of the History of Linguistic Phonology".
