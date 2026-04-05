# ADR-001: Why MLP for Level A (ABCs)

**Status:** Accepted  
**Date:** 2026-04-04  
**Context:** Level A teaches letter recognition (26 classes) and letter-to-phoneme mapping (26 → 44).

## Decision

Use a Multi-Layer Perceptron (feed-forward neural network) for Level A.

## Rationale

1. **The problem is classification.** 26 letters, one-hot vectors. Simplest neural network task.

2. **Right-sized tool.** Transformer attention adds zero benefit for fixed-input classification. RNN recurrence adds nothing with no sequence.

3. **Maximally educational.** Teaches fundamentals without architectural complexity.

4. **Size target.** ~3,400 parameters = ~13.7KB. Fits on ESP32.

## Alternatives Considered

- **Lookup table:** Works but teaches nothing about neural networks.
- **Transformer:** Overkill. No sequence, no attention needed.
- **CNN:** Could work for image recognition, but we're using one-hot encoding.

## Consequences

- Models are tiny (<100KB), deploy anywhere
- Students learn fundamentals before complexity
- MLP foundation (Layer, Network, Trainer) reused in all subsequent levels
