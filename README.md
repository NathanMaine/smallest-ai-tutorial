
![Gemini_Generated_Image_h9yav1h9yav1h9ya](https://github.com/user-attachments/assets/8f152eb3-4254-4fe0-a246-9f9086a93b61)

# Build 4 Neural Networks from Scratch in Pure Python

**No NumPy. No PyTorch. No magic. Just you, a text editor, and matrix multiplication.**

I built four neural networks from scratch — no ML libraries, no automatic differentiation, not even NumPy. Just pure Python lists and arithmetic. Along the way I learned more about how these things *actually work* than I had in years of using the high-level frameworks.

This repo is the tutorial version: everything cleaned up, annotated, and structured so you can build them too.

---

## What You'll Build

Each tutorial takes you through one architecture, applied to a single domain: **English phonics** (teaching a computer which sounds the letters make).

Why phonics? It's a domain with natural complexity. Single letters are easy for an MLP. Letter combinations need sequence memory. Context-dependent rules benefit from attention. It scales nicely from "trivial" to "genuinely interesting" as the architectures get more powerful.

| Level | Architecture | Domain | Parameters | Accuracy |
|-------|-------------|--------|-----------|----------|
| 01 | **MLP** (Multilayer Perceptron) | Single letter → sound | ~3,400 | 95%+ |
| 02 | **LSTM** (Long Short-Term Memory) | Letter sequences → phonemes | ~12,000 | 88%+ |
| 03 | **Transformer** | Context-aware reading | ~45,000 | 92%+ |
| 04 | **Comparison Study** | MoE, Mamba, BitNet, quantization | varies | — |

---

## Why Build from Scratch?

You can use PyTorch in one line. So why do this the hard way?

Because when things go wrong in a real model, "it's in the framework somewhere" is not a useful answer. Understanding backpropagation at the level of "which weight got which gradient and why" is different from trusting that `loss.backward()` did something reasonable.

Building from scratch also builds the right intuitions. After implementing a forward pass by hand, you genuinely understand why batching matters, why initialization is subtle, and why the chain rule can both save you and destroy you.

You don't have to do this forever. But doing it once, deeply, changes how you read papers and debug models.

---

## What You'll Learn

By the end of all four tutorials:

- How dot products and matrix multiplication form the *only* computation neural networks do
- Why activation functions are necessary and what happens without them
- How backpropagation actually works (the chain rule, spelled out)
- Why vanilla RNNs struggle with long sequences (and the vanishing gradient problem)
- How LSTM gates solve that problem with learned memory
- Why attention is more powerful than recurrence for many tasks
- What "ternary weights" mean and why BitNet can be 20x smaller

---

## Prerequisites

- **Python 3.10+** — that's it for dependencies
- **Basic algebra** — you should be comfortable with vectors and matrices at a conceptual level (not calculus-fluent, just not afraid)
- **Curiosity** — seriously, this is the most important one

You do not need to have built a neural network before. Tutorial 01 starts from scratch.

---

## Getting Started

```bash
git clone https://github.com/your-username/smallest-ai-tutorial
cd smallest-ai-tutorial
```

No `pip install` needed. No virtual environment. No CUDA drivers.

To verify everything works:

```bash
python3 -m pytest tutorials/01-mlp-from-scratch/tests/ -v
```

Then head to [SETUP.md](SETUP.md) for a complete walkthrough.

---

## Tutorial Structure

Each tutorial (`01`, `02`, `03`, `04`) has the same layout:

```
tutorials/01-mlp-from-scratch/
├── README.md          — The chapter narrative. Read this first.
├── lesson.md          — Detailed walkthrough with code explained line by line
├── starter_code/      — Skeleton files: function signatures + "raise NotImplementedError"
├── solution/          — Complete working implementations
└── tests/             — pytest tests to verify your implementation
```

**Recommended flow:**

1. Read `README.md` (the "why")
2. Read `lesson.md` (the "how")
3. Try implementing from `starter_code/`
4. Run the tests to check your work
5. Compare to `solution/` when stuck

---

## About the Code Style

Files are numbered `01_`, `02_`, etc. Python doesn't allow `import 01_math_foundations` directly (identifiers can't start with digits), so the files use `importlib.import_module('01_math_foundations')`. Each file explains this when it appears.

Each solution file is a standalone script you can run directly:

```bash
python3 tutorials/01-mlp-from-scratch/solution/01_math_foundations.py
```

It will print a demonstration of everything in that chapter.

---

## A Note on the Domain

The phonics domain is real. The data in `data/phonics/` reflects actual English phonics rules — CVC words, digraphs, vowel sounds. A trained model can look at a letter pattern and predict its sound, which is genuinely useful for reading instruction.

We use phonics not because it's the flashiest application, but because it gives us a clean problem with just enough complexity to motivate each architectural upgrade.

---

## Contributing

Found a bug? Got a clearer explanation? PRs are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

*Start with [Tutorial 01: MLP from Scratch](tutorials/01-mlp-from-scratch/README.md)*
