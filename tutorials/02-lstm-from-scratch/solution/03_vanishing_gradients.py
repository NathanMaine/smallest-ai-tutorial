"""
Chapter 3 — Vanishing Gradients
================================

Why vanilla RNNs fail on long sequences, why tanh saturates,
and the motivation for LSTM.

What this module teaches
-------------------------
In Chapter 2 we built a complete vanilla RNN with BPTT. The backward pass
works — but there is a fundamental problem hiding in the math: gradients
*vanish* as they flow backward through time.

Why gradients vanish
---------------------
At every time step, the gradient dh_next is transformed by two operations:

    dh_raw = dh * tanh'(h_t)          # tanh derivative, always in [0, 1]
    dh_prev = W_hidden^T @ dh_raw     # matrix multiply

The tanh derivative is:
    tanh'(x) = 1 - tanh(x)^2

When |h_t| is near 1 (saturated), tanh'(h_t) ≈ 0.
When |h_t| is near 0, tanh'(h_t) ≈ 1.

Even in the best case, if the spectral radius of W_hidden (largest singular
value) is < 1, repeated multiplication shrinks the gradient exponentially.
Over 20–50 steps the gradient reaching early time steps is effectively zero.

Consequence
-----------
The RNN cannot learn long-term dependencies. It has no mechanism to
"decide" that a gradient should be preserved. LSTM solves this with
multiplicative gates: a forget gate can hold the gradient at exactly 1.0,
preventing decay.

This chapter
------------
We measure gradient magnitude at each time step during BPTT to make the
vanishing phenomenon concrete and visible.

Chapter roadmap
---------------
  Chapter 1: Recurrence — hidden state, SimpleMemoryCell
  Chapter 2: Vanilla RNN — forward through time and BPTT
  Chapter 3: Vanishing gradients — why vanilla RNNs can't learn long-term
             dependencies  ← you are here
  Chapter 4: LSTM — gated memory, forget/input/output gates
  Chapter 5: Attention — weighting past states by relevance
"""

import importlib
import sys
import os
import math
import random

# ---------------------------------------------------------------------------
# Import from previous chapters
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(__file__))

rnn_mod = importlib.import_module('02_vanilla_rnn')
RNN = rnn_mod.RNN

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax


# ---------------------------------------------------------------------------
# Helper: vector L2 norm
# ---------------------------------------------------------------------------

def _vec_norm(v):
    """Return the L2 norm (magnitude) of vector v."""
    return math.sqrt(sum(x * x for x in v))


# ---------------------------------------------------------------------------
# measure_gradient_flow
# ---------------------------------------------------------------------------

def measure_gradient_flow(seq_length, hidden_size, seed=42):
    """
    Measure how the purely-propagated gradient decays across time steps.

    The vanishing gradient problem is about the gradient that *travels backward
    through time* — the dh_next signal passed from step t+1 to step t through
    the recurrent weight matrix and the tanh derivative. Each fresh output loss
    at time t adds new gradient signal, masking the decay. This function
    isolates the pure propagation effect by injecting a single loss only at the
    LAST time step and then tracking how that gradient shrinks as it travels
    backward step by step.

    Key observation:
      - For short sequences, the gradient stays reasonably large at all steps.
      - For long sequences, the gradient decays at each step (often to near
        zero at the earliest steps) because tanh' <= 1 and W_hidden repeatedly
        multiplies values that can be < 1.

    Parameters
    ----------
    seq_length  : int — number of time steps in the sequence
    hidden_size : int — dimensionality of the RNN hidden state
    seed        : int — random seed for reproducibility

    Returns
    -------
    list[float]
        Gradient ratios of length (seq_length - 1).
        ratio[t] = dh_next_norm_at_step_t / dh_next_norm_at_step_(T-2)

        Index 0 corresponds to the *earliest* propagated gradient
        (after travelling through seq_length - 1 steps of decay).
        Index -1 is the norm after a single propagation step.

        Values < 1 mean the gradient shrank relative to the step just before
        the final step. Values approaching 0 indicate vanishing gradients.
    """
    rng = random.Random(seed)

    # Build RNN with 3-dimensional inputs and outputs
    input_size = 3
    output_size = 3
    net = RNN(input_size=input_size, hidden_size=hidden_size,
              output_size=output_size, seed=seed)

    # Generate a random sequence
    sequence = [
        [rng.random(), rng.random(), rng.random()]
        for _ in range(seq_length)
    ]

    # Only one target: we inject loss at the LAST time step only.
    # This isolates the pure propagation: the gradient at each earlier step
    # is entirely the result of the signal travelling backward through the
    # recurrent connections, with NO fresh injection.
    last_target = rng.randint(0, output_size - 1)

    # -----------------------------------------------------------------------
    # Run forward pass to populate caches
    # -----------------------------------------------------------------------
    net._zero_all_gradients()
    net.forward(sequence)

    # -----------------------------------------------------------------------
    # Custom backward pass: inject loss only at t = T-1, then propagate back.
    # Record the norm of dh_next at each step.
    # -----------------------------------------------------------------------
    T = seq_length

    # Start: compute gradient at the last time step
    probs = softmax(net.outputs[T - 1])
    one_hot = [0.0] * output_size
    one_hot[last_target] = 1.0
    dout = [probs[i] - one_hot[i] for i in range(output_size)]

    # dh from output layer at t=T-1: W_output^T @ dout
    dh = [0.0] * hidden_size
    for j in range(hidden_size):
        for i in range(output_size):
            dh[j] += net.W_output[i][j] * dout[i]

    # Through tanh at t=T-1
    h_T = net.hiddens[T]  # hidden state at last step
    dh_raw = [dh[j] * (1.0 - h_T[j] * h_T[j]) for j in range(hidden_size)]

    # Propagate one step back: dh_next now holds the gradient that will enter t=T-2
    dh_next = [0.0] * hidden_size
    for j in range(hidden_size):
        for i in range(hidden_size):
            dh_next[j] += net.cell.W_hidden[i][j] * dh_raw[i]

    # dh_next_norms[k] = norm of dh_next after k propagation steps
    # k=1 means the norm after propagating through 1 step (entering t=T-2)
    # k=seq_length-1 means after propagating through all steps (entering t=0 from left)
    dh_next_norms = []

    # Record norm after first propagation (entering t = T-2)
    dh_next_norms.append(_vec_norm(dh_next))

    # Continue propagating backward through steps T-2 down to 1
    # (recording norm as the signal passes through each additional step)
    for t in range(T - 2, 0, -1):
        h_t = net.hiddens[t + 1]
        # No output loss here — pure propagation
        dh_raw = [dh_next[j] * (1.0 - h_t[j] * h_t[j]) for j in range(hidden_size)]
        dh_next = [0.0] * hidden_size
        for j in range(hidden_size):
            for i in range(hidden_size):
                dh_next[j] += net.cell.W_hidden[i][j] * dh_raw[i]
        dh_next_norms.append(_vec_norm(dh_next))

    # dh_next_norms is ordered from shallowest propagation to deepest:
    #   [1-step, 2-step, ..., (T-1)-step]
    # index 0 = gradient after 1 propagation step (entered t=T-2)
    # index -1 = gradient after T-1 steps (reaching t=0 from the right)
    #
    # We want: ratio[i] = norms[i] / norms[0]
    # so that ratio[0] = 1.0 (the shallowest, largest step) and
    # ratio[-1] shows how much the gradient shrank over the full depth.
    # But the spec asks: ratio[0] is the earliest step, ratio[-1] is the
    # step just before the end.  So we reverse:
    #   reversed: [deepest, ..., shallowest]
    #   ratio[0] = deepest / shallowest → should be small for long seqs
    #   ratio[-1] = shallowest / shallowest = 1.0 (by definition)
    #
    # We normalise by the LAST entry (shallowest = index 0 of original,
    # index -1 of reversed).

    dh_next_norms.reverse()
    # Now dh_next_norms[0] = deepest (earliest step), [-1] = shallowest (latest step)

    ref = dh_next_norms[-1]  # shallowest propagation norm (closest to output)
    if ref == 0.0:
        return [1.0] * (seq_length - 1)

    ratios = [n / ref for n in dh_next_norms]
    return ratios


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 02-lstm-from-scratch/solution/03_vanishing_gradients.py
# ---------------------------------------------------------------------------

def _bar(value, max_value, width=40):
    """Render a text bar chart segment."""
    if max_value == 0:
        filled = 0
    else:
        filled = int(round(value / max_value * width))
    filled = max(0, min(filled, width))
    return '#' * filled + '.' * (width - filled)


if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 3 — Vanishing Gradients")
    print("Demonstrating why vanilla RNNs struggle with long sequences")
    print("=" * 65)

    print("""
Theory recap
------------
At each BPTT step the gradient is multiplied by:
  tanh'(h_t) = 1 - h_t^2   [always in [0, 1]]
  W_hidden^T               [can shrink or amplify]

If the dominant eigenvalue of W_hidden is < 1, each step shrinks the
gradient. Over many steps the product approaches zero — the gradient
at early time steps effectively disappears.
""")

    for seq_len in [5, 20, 50]:
        ratios = measure_gradient_flow(seq_length=seq_len, hidden_size=8, seed=42)
        max_ratio = max(ratios) if ratios else 1.0

        print(f"Sequence length = {seq_len}  (hidden_size=8)")
        print(f"  {'Step':>5}  {'Ratio':>8}  Gradient bar (relative to last step)")
        print(f"  {'-'*5}  {'-'*8}  {'-'*42}")

        # Print each time step
        for t, r in enumerate(ratios):
            bar = _bar(r, max_ratio)
            marker = " ← earliest" if t == 0 else ""
            print(f"  {t:>5}  {r:>8.4f}  {bar}{marker}")

        # Summarise vanishing
        early_avg = sum(ratios[:max(1, len(ratios) // 4)]) / max(1, len(ratios) // 4)
        late_avg  = sum(ratios[-(max(1, len(ratios) // 4)):]) / max(1, len(ratios) // 4)
        print(f"\n  Early steps avg ratio : {early_avg:.4f}")
        print(f"  Late  steps avg ratio : {late_avg:.4f}")
        if early_avg < late_avg * 0.5:
            print("  → Clear vanishing: early gradients are < 50% of late gradients")
        else:
            print("  → Gradients relatively stable (short sequence, not yet vanishing)")
        print()

    print("=" * 65)
    print("Observation: for length-50 sequences, the earliest time steps")
    print("receive nearly zero gradient — the RNN cannot learn from them.")
    print()
    print("Solution: LSTM (Chapter 4) uses a *forget gate* that can hold")
    print("gradient magnitude at 1.0, allowing gradients to flow unchanged")
    print("across hundreds of steps.")
    print("=" * 65)
