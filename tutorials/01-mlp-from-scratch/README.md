# Tutorial 01: MLP from Scratch

**Build a Multilayer Perceptron in pure Python. No libraries.**

---

## What Makes This Tricky

The thing that surprised me most when building an MLP from scratch wasn't the math. The math is manageable. What tripped me up was how much of PyTorch's "magic" is actually just bookkeeping.

During the forward pass, you have to *remember* your intermediate values (the pre-activation `z` and the post-activation `a` for each layer) because the backward pass needs them. If you throw them away, you have to re-run the forward pass during backpropagation, which defeats the purpose.

The second gotcha: backpropagation is the chain rule applied backwards through a list. That's it. But the indexing is fiddly, and off-by-one errors are invisible until your loss doesn't decrease.

---

## What Surprised Me

After implementing the training loop and watching the loss actually decrease, I felt something I didn't expect: relief. Not pride — *relief*. Because for the first time I was certain that the thing I was building was doing what I thought it was doing. No black box.

Also: a 3,400-parameter network trained for 500 epochs on 26 examples fits in about 14 KB and classifies all 26 letters with 95%+ accuracy. That's smaller than most JPEG images.

---

## What You'll Build

A complete MLP that:
- Takes a 26-dimensional one-hot vector (representing a letter) as input
- Passes it through a hidden layer of 64 ReLU neurons
- Produces a 26-dimensional output
- Is trained with SGD + cross-entropy loss + backpropagation
- Achieves 95%+ accuracy at classifying all 26 letters

Seven chapters, building one piece at a time.

---

## Chapter Overview

| Chapter | File | What It Covers |
|---------|------|----------------|
| 1 | `01_math_foundations.py` | Vectors, matrices, dot product — the atoms |
| 2 | `02_single_neuron.py` | One neuron: weighted sum + activation |
| 3 | `03_forward_pass.py` | Layer and Network: many neurons, chained |
| 4 | `04_loss_function.py` | Softmax, cross-entropy, MSE |
| 5 | `05_backpropagation.py` | The chain rule, spelled out |
| 6 | `06_training_loop.py` | SGD: forward → loss → backward → update |
| 7 | `07_final_project.py` | LetterClassifier: the first complete AI |

---

## How to Work Through It

**Option A: Just read the solutions**
Run each solution file and read the output. Good for building intuition before you implement.

```bash
python3 tutorials/01-mlp-from-scratch/solution/01_math_foundations.py
```

**Option B: Implement from starter code**
Open `starter_code/01_math_foundations.py`, read the docstrings, implement each function. Run the tests to check.

```bash
python3 -m pytest tutorials/01-mlp-from-scratch/tests/test_01_math.py -v
```

**Option C: Read lesson.md first**
`lesson.md` has a detailed walkthrough of each chapter with code explained line by line. Read it, then implement.

---

## Recommended Reading Order

1. This file (you're reading it)
2. [lesson.md](lesson.md) — the narrative
3. `starter_code/01_math_foundations.py` — implement it
4. Run `tests/test_01_math.py` — check your work
5. Repeat for chapters 2-7
6. Run `python3 solution/07_final_project.py` — see the whole thing work

---

## Running the Complete Solution

The final project is self-contained:

```bash
python3 tutorials/01-mlp-from-scratch/solution/07_final_project.py
```

This will train the letter classifier for 500 epochs and print accuracy for all 26 letters. Takes about 5-10 seconds.

---

## What's Next

Once you've got the MLP working, Tutorial 02 introduces **LSTMs** — networks with memory. The motivating question: "Our MLP handles one letter at a time. What if we need to look at a sequence of letters to figure out how the first one sounds?"

→ [Tutorial 02: LSTM from Scratch](../02-lstm-from-scratch/README.md)
