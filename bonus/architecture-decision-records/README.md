# Architecture Decision Records

These documents explain *why* each architectural choice was made across the four tutorials. They're useful if you want to understand the reasoning, not just the implementation.

---

## What Is an ADR?

An Architecture Decision Record is a short document that captures:
- The decision made
- The context that led to it
- The alternatives that were considered
- The consequences

They're common in software engineering. Here we use them to explain neural network architecture choices.

---

## Documents

| ADR | Decision |
|-----|----------|
| [ADR-001](001-why-mlp-for-abcs.md) | Why MLP for Tutorial 01 (single-letter classification) |
| [ADR-002](002-why-rnn-for-phonics.md) | Why RNN/LSTM for Tutorial 02 (letter sequences) |
| [ADR-003](003-why-transformer-for-reading.md) | Why Transformer for Tutorial 03 (context-dependent rules) |
| [ADR-004](004-level-d-comparison-study.md) | How the four architectures in Tutorial 04 were selected |

---

## Reading Order

If you've completed all four tutorials and want to understand the design, read these in order. They make more sense after you've seen the architectures in action.
