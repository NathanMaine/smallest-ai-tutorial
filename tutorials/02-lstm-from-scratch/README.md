# Tutorial 02: LSTM from Scratch

**Build a Long Short-Term Memory network — the architecture that gave AI its memory.**

---

## What Makes This Tricky

The MLP from Tutorial 01 is stateless. Feed it 'c' and it predicts /k/. Feed it 'c' again after reading "ch" and it still predicts /k/. It has no memory of what came before.

That's a real problem. In "chip", the 'c' makes a /tʃ/ sound because of the 'h' that follows. In "cat", it makes /k/. Same letter, different context, different sound. An MLP can't learn this.

The tricky part of RNNs isn't the forward pass — it's understanding why vanilla RNNs fail to remember across long sequences (the vanishing gradient problem), and then understanding why LSTM's gates solve it. The math is genuinely subtle.

The other tricky thing: backpropagation through time (BPTT). You have to unroll the recurrent computation across time steps and compute gradients at each one. The bookkeeping is more complex than in the MLP.

---

## What Surprised Me

The forget gate bias initialization. All biases in the LSTM start at 0 — except the forget gate bias, which starts at 1.0. This means the network begins by remembering everything, then *learns what to forget*. Starting at 0 (which means sigmoid(0) = 0.5, discarding half the cell state immediately) makes training much harder. This isn't obvious from reading the equations — it's one of those things you only learn by reading the paper carefully (Jozefowicz et al., 2015).

Also: the LSTM demo in Chapter 4 shows RNN vs LSTM memory retention over time. Watching the RNN's signal decay exponentially while the LSTM holds it stable is genuinely satisfying.

---

## What You'll Build

A complete LSTM that:
- Processes variable-length sequences of letter one-hot vectors
- Maintains hidden state and cell state across time steps
- Has forget, input, and output gates
- Is trained with Backpropagation Through Time (BPTT)
- Can blend CVC word letter sequences into phoneme sequences

Six chapters, building the recurrent machinery piece by piece.

---

## Chapter Overview

| Chapter | File | What It Covers |
|---------|------|----------------|
| 1 | `01_recurrence.py` | Hidden state, tanh, SimpleMemoryCell |
| 2 | `02_vanilla_rnn.py` | Full RNN with BPTT |
| 3 | `03_vanishing_gradients.py` | Why vanilla RNNs fail on long sequences |
| 4 | `04_lstm_cell.py` | LSTM: forget, input, output gates |
| 5 | `05_sequence_model.py` | Trainable LSTM with full BPTT |
| 6 | `06_final_project.py` | PhonicsBlender: CVC letter → phoneme |

---

## Prerequisites

Tutorial 01 (MLP from Scratch). The LSTM tutorials import math functions and activations from Tutorial 01's solution directory.

---

## Running the Complete Solution

```bash
python3 tutorials/02-lstm-from-scratch/solution/06_final_project.py
```

This trains an LSTM on CVC word data and shows phoneme predictions for each letter position.

---

## Why LSTM After MLP?

The MLP handles one letter in isolation. The LSTM handles sequences. This is the first time our model can use context — the letter before (and after, if we go bidirectional) influences the prediction.

The motivation becomes clear in Chapter 3 (`03_vanishing_gradients.py`), which runs a demonstration showing gradient magnitudes decaying to near-zero as we backpropagate through 50+ time steps of a vanilla RNN. Then Chapter 4 shows LSTM gates solving this.

---

## What's Next

After LSTM, we introduce the **Transformer** — an architecture that doesn't use recurrence at all. Instead, it uses attention to let every position directly look at every other position. This turns out to be faster to train and often more powerful.

→ [Tutorial 03: Transformer from Scratch](../03-transformer-from-scratch/README.md)
