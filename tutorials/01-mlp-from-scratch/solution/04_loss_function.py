"""
Chapter 4 — Loss Functions
==========================

How we measure "how wrong" the model is.

Before a model can learn, it needs a single number that summarises its
mistake on a given example. That number is the *loss* (also called *cost*
or *error*). Training is the process of driving this number toward zero.

  Softmax        — converts a vector of raw scores (logits) into a proper
                   probability distribution: all values > 0, sum = 1.0.

  Cross-entropy  — the standard loss for classification. It compares the
                   model's probability distribution to the true one-hot
                   target and returns a non-negative scalar; zero only when
                   predicted == target perfectly.

  MSE            — Mean Squared Error: the standard loss for regression.
                   Averages the squared difference between each predicted
                   and target value.

Key ideas
---------
  Numerical stability — softmax subtracts max(logits) before calling exp().
  This prevents overflow (e.g. exp(1000)) without changing the output,
  because the constant cancels in the ratio.

  Log-clamp — cross-entropy calls log(predicted). If predicted is exactly 0,
  log(0) = -inf. A tiny epsilon (1e-15) clamps predicted away from zero.

Chapter roadmap
---------------
  Chapter 1:  Math foundations — vectors & matrices
  Chapter 2:  Single neuron — dot product + bias + activation
  Chapter 3:  Forward pass — Layer and Network classes
  Chapter 4:  Loss functions — softmax, cross-entropy, MSE  ← you are here
  Chapter 5:  Backpropagation — gradients via the chain rule
  Chapter 6:  Training loop — putting it all together
"""

import math

# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def softmax(logits):
    """Convert a list of raw scores (logits) to a probability distribution.

    Each output is in (0, 1) and the outputs sum to exactly 1.0.

    Numerical stability trick: subtract max(logits) before calling exp().
    This keeps all exponent arguments <= 0, preventing overflow.
    The result is identical to the naïve formula because the constant
    cancels in the numerator and denominator.

        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Parameters
    ----------
    logits : list[float] — raw unnormalised scores, any real values

    Returns
    -------
    list[float] — same length as logits, values in (0, 1), sums to 1.0
    """
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Cross-entropy loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(predicted, target):
    """Compute the cross-entropy between a predicted distribution and target.

    Formula:  L = -sum(target_i * log(predicted_i))

    For one-hot targets this reduces to -log(predicted[true_class]).
    The loss is zero when the model assigns probability 1.0 to the correct
    class, and grows toward infinity as the predicted probability falls.

    A small epsilon (1e-15) clamps predicted values away from zero to
    prevent log(0) = -inf.

    Parameters
    ----------
    predicted : list[float] — model's probability distribution (should sum to 1)
    target    : list[float] — true distribution, usually one-hot

    Returns
    -------
    float — non-negative scalar; lower is better
    """
    epsilon = 1e-15
    loss = 0.0
    for p, t in zip(predicted, target):
        p_clamped = max(p, epsilon)
        loss -= t * math.log(p_clamped)
    return loss


# ---------------------------------------------------------------------------
# MSE loss
# ---------------------------------------------------------------------------

def mse_loss(predicted, target):
    """Compute the Mean Squared Error between predicted and target values.

    Formula:  MSE = (1/n) * sum((predicted_i - target_i)^2)

    Zero when predicted == target exactly. Penalises large errors more
    heavily than small ones (quadratic, not linear).

    Parameters
    ----------
    predicted : list[float] — model's output values
    target    : list[float] — ground-truth values (same length)

    Returns
    -------
    float — non-negative scalar; lower is better
    """
    n = len(predicted)
    return sum((p - t) ** 2 for p, t in zip(predicted, target)) / n


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 phase1-from-scratch/level-a-abcs/04_loss_function.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4 — Loss Functions Demo")
    print("Softmax, Cross-Entropy, and MSE from scratch")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Demo 1: Softmax
    # -----------------------------------------------------------------------
    print("\n--- Softmax ---")
    logits = [2.0, 1.0, 0.5]
    probs = softmax(logits)
    print(f"logits:         {logits}")
    print(f"probabilities:  {[round(p, 6) for p in probs]}")
    print(f"sum:            {round(sum(probs), 10)}  (should be 1.0)")

    print("\nNumerical stability test (logits with large values):")
    big_logits = [1000.0, 1001.0, 1002.0]
    big_probs = softmax(big_logits)
    print(f"logits: {big_logits}")
    print(f"probs:  {[round(p, 6) for p in big_probs]}")
    print(f"sum:    {round(sum(big_probs), 10)}  (no overflow — stable!)")

    # -----------------------------------------------------------------------
    # Demo 2: Cross-entropy — good prediction
    # -----------------------------------------------------------------------
    print("\n--- Cross-Entropy Loss ---")
    target = [0.0, 0.0, 1.0]   # true class is index 2

    predicted_good = [0.01, 0.01, 0.98]
    loss_good = cross_entropy_loss(predicted_good, target)
    print(f"target:           {target}")
    print(f"good prediction:  {predicted_good}")
    print(f"loss (good):      {round(loss_good, 6)}  (should be < 0.1)")

    predicted_bad = [0.98, 0.01, 0.01]
    loss_bad = cross_entropy_loss(predicted_bad, target)
    print(f"\nbad prediction:   {predicted_bad}")
    print(f"loss (bad):       {round(loss_bad, 6)}  (should be > 3.0)")

    # -----------------------------------------------------------------------
    # Demo 3: MSE
    # -----------------------------------------------------------------------
    print("\n--- MSE Loss ---")
    p_perfect = [1.0, 0.0, 0.0]
    t_perfect = [1.0, 0.0, 0.0]
    print(f"predicted: {p_perfect}")
    print(f"target:    {t_perfect}")
    print(f"MSE:       {mse_loss(p_perfect, t_perfect)}  (perfect = 0)")

    p_off = [1.0, 0.0]
    t_off = [0.0, 1.0]
    print(f"\npredicted: {p_off}")
    print(f"target:    {t_off}")
    print(f"MSE:       {mse_loss(p_off, t_off)}  (worst binary case = 1.0)")

    print("\n" + "=" * 60)
    print("Chapter 4 complete. Chapter 5: Backpropagation.")
    print("=" * 60)
