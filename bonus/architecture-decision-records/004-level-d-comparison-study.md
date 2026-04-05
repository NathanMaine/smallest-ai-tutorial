# ADR 004 — Level D: Architecture Comparison Study Approach

**Status:** Accepted  
**Date:** 2026-04-04  
**Deciders:** Nathan Maine

---

## Context

Levels A, B, and C each demonstrated one neural network architecture:

- Level A (MLP) → letter-to-phoneme mapping
- Level B (RNN/LSTM) → CVC and digraph sequences
- Level C (Transformer) → reading-level sentence phonics

Each level made a motivated choice. ADR 001, 002, and 003 document the reasoning. Those choices produced working systems, but the methodology has a gap: the architectures were never compared head-to-head on the same task. It is unknown whether the chosen architecture actually outperforms alternatives or whether the task is simply easy enough that any reasonable model succeeds.

---

## Decision

Level D runs a **controlled comparison study** across five core architectures (MLP, vanilla RNN, LSTM, Transformer, 1-D CNN) plus an ensemble, all trained on a single unified dataset derived from all previous levels.

The dataset is held constant. Only the architecture varies.

---

## Rationale

### Why a comparison study instead of a new task?

The most common failure mode in "build a neural net from scratch" projects is confirmation bias: the developer picks an architecture they expect to work, it works, and no alternative is ever tested. A comparison study forces the question: *what is this architecture actually better at?*

### Why the same unified dataset?

Mixing data from all previous levels creates a multi-type problem that is not trivially easy for any single architecture:

- Letter-only examples (1-char input, 1-phoneme output) favor MLP
- CVC/digraph words (3-4 char input, 3-4 phoneme output) favor RNN and CNN
- Silent-e words (long-range dependency: final 'e' affects earlier vowel) favor LSTM and Transformer
- Vowel team words (digraph input tokens) test tokenization sensitivity

No architecture has a natural advantage on all types simultaneously.

### Why no hyperparameter tuning per architecture?

Tuning each architecture separately would optimize for the dataset, defeating the purpose. The goal is to observe *default* inductive biases, not to find the best configuration for each model.

### Why include CNN?

1-D CNNs are rarely discussed in phonics/NLP teaching material despite being highly effective at local pattern detection. Digraphs ("sh", "ch", "th") and consonant blends are exactly the kind of fixed-width local patterns CNNs are designed to detect. Including CNN surfaces this.

### Why include an ensemble?

The ensemble chapter serves two purposes:

1. Establishes a performance ceiling — the best achievable result when all architectures collaborate
2. Reveals which architectures contribute unique signal vs. which are redundant

---

## Consequences

### Positive

- Empirical evidence replaces assumption about which architecture is appropriate
- Developers reading the code learn *why* architecture choice matters, not just *that* it matters
- The comparison creates reusable infrastructure (unified dataset, shared evaluation) for future levels

### Negative

- Level D is wider than earlier levels (8 chapters vs. 5-6)
- Running all architectures takes longer than a single-architecture level
- The "no hyperparameter tuning" constraint means individual architectures will not perform at their theoretical best — some readers may find this unsatisfying

### Neutral

- Results will vary with random seed; the comparison is intended to reveal structural differences, not to declare a definitive winner
- The dataset is intentionally small (~129 examples); results at this scale may not generalize to production-size data

---

## Alternatives Considered

**Alternative 1: Add one new architecture per level**  
Rejected — makes the architectural comparison spread across levels with different datasets, preventing direct comparison.

**Alternative 2: Use only the hardest subset (silent-e + vowel teams)**  
Rejected — too few examples (~22) to produce meaningful accuracy differences; overfitting dominates.

**Alternative 3: Introduce a new phonics task for Level D**  
Rejected — complicates the study with task confounds. The goal is to isolate architecture, not introduce new data challenges.

---

## Related Decisions

- ADR 001: Why MLP for letter-to-phoneme (Level A)
- ADR 002: Why RNN for CVC and digraph sequences (Level B)
- ADR 003: Why Transformer for reading-level phonics (Level C)
