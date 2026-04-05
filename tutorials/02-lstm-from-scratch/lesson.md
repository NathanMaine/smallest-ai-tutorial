# Lesson: LSTM from Scratch

The detailed walkthrough. Read this alongside the code.

---

## Chapter 1: Recurrence — The Idea of Memory

An MLP is stateless. Every forward pass is independent — the same input always produces the same output. That's fine for classifying a single letter. It's a problem for sequences.

The solution is a **hidden state**: a vector that gets updated at every time step. The RNN equation is:

```
h_t = tanh(W_input @ x_t + W_hidden @ h_{t-1} + bias)
```

The same input `x = [1, 0]` can produce different outputs depending on `h_{t-1}` — the history. Chapter 1 demonstrates this with a `SimpleMemoryCell`.

### Why tanh?

Sigmoid outputs (0, 1). Tanh outputs (-1, 1) and is zero-centered. Zero-centered outputs mean gradients don't all have the same sign, which makes training more stable. Tanh also saturates more gracefully than sigmoid for large inputs.

---

## Chapter 2: Vanilla RNN and BPTT

A vanilla RNN unrolls through time. For a sequence of length T, we run T forward steps, then backpropagate through all T time steps. This is **Backpropagation Through Time (BPTT)**.

The problem: at each time step, the gradient is multiplied by the weight matrix `W_hidden`. If the weights are small, the gradient shrinks exponentially (vanishing). If the weights are large, it grows exponentially (exploding).

---

## Chapter 3: Vanishing Gradients

This chapter is a demonstration, not an implementation. Run `03_vanishing_gradients.py` and watch the gradient magnitude at time step 1 compared to time step 50. With a vanilla RNN, the ratio falls to nearly zero for long sequences.

This is the problem that LSTM solves.

---

## Chapter 4: The LSTM Cell

LSTM has four things instead of one:

| Component | Formula | Purpose |
|-----------|---------|---------|
| Forget gate | `f = sigmoid(W_f @ [x, h] + b_f)` | What to erase from cell state |
| Input gate | `i = sigmoid(W_i @ [x, h] + b_i)` | What new info to write |
| Candidate | `c_hat = tanh(W_c @ [x, h] + b_c)` | The new content to potentially write |
| Output gate | `o = sigmoid(W_o @ [x, h] + b_o)` | What to read from cell state |

The cell state update:
```
c_t = f * c_{t-1} + i * c_hat
h_t = o * tanh(c_t)
```

The critical insight: `c_t` is updated by *element-wise multiplication* (not matrix multiplication). This means gradients can flow through the cell state highway without the exponential decay problem.

### The Forget Bias = 1.0

This is the detail most implementations get wrong. The forget gate bias starts at 1.0 (not 0.0). sigmoid(1.0) ≈ 0.73, which means the gate starts "mostly open" — the network begins by remembering most things. Starting at 0 means sigmoid(0) = 0.5, which discards half the cell state from step one and makes early training hard.

---

## Chapter 5: Trainable LSTM with BPTT

Now we add the backward pass. BPTT through LSTM gates requires computing gradients for all four gates at each time step, then propagating backwards through the cell state highway.

The key equation for the cell state gradient:
```
dc_{t-1} = dc_t * f_t
```

When `f_t` is close to 1 (the network chose to remember), the gradient passes through nearly unchanged. This is why LSTM can learn dependencies over hundreds of time steps.

---

## Chapter 6: PhonicsBlender — First Sequence Model

The PhonicsBlender applies the LSTM to a real problem: given a CVC word letter by letter (c-a-t), predict the phoneme at each position (/k/ /æ/ /t/).

This is **sequence-to-sequence**: input length equals output length. For CVC words, every letter has a corresponding phoneme target.

After training, the model achieves high accuracy because CVC words have regular phonics patterns — each letter's sound is mostly determined by the vowel type and position, which the LSTM's hidden state captures.
