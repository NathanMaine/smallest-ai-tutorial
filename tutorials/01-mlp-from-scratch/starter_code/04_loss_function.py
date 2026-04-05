"""
Chapter 4 — Loss Functions
==========================

How we measure "how wrong" the model is.

Before a model can learn, it needs a single number that summarises its
mistake on a given example. That number is the *loss*. Training is the
process of driving this number toward zero.

  Softmax        — converts raw scores (logits) into a probability distribution:
                   all values > 0, sum = 1.0.

  Cross-entropy  — the standard loss for classification. Compares the model's
                   distribution to the true one-hot target.

  MSE            — Mean Squared Error: the standard loss for regression.

Key idea — numerical stability:
  Softmax subtracts max(logits) before calling exp() to prevent overflow.
  Cross-entropy clamps predicted probabilities away from 0 to prevent log(0) = -inf.
"""

import math


def softmax(logits):
    """Convert a list of raw scores (logits) to a probability distribution.

    Each output is in (0, 1) and the outputs sum to exactly 1.0.

    Numerical stability trick: subtract max(logits) before calling exp().
    This keeps all exponent arguments <= 0, preventing overflow.

        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Parameters
    ----------
    logits : list[float] — raw unnormalised scores, any real values

    Returns
    -------
    list[float] — same length as logits, values in (0, 1), sums to 1.0
    """
    raise NotImplementedError(
        "Your turn!\n"
        "  1. Find max_val = max(logits)\n"
        "  2. Compute exps = [math.exp(x - max_val) for x in logits]\n"
        "  3. total = sum(exps)\n"
        "  4. Return [e / total for e in exps]"
    )


def cross_entropy_loss(predicted, target):
    """Compute the cross-entropy between a predicted distribution and target.

    Formula:  L = -sum(target_i * log(predicted_i))

    For one-hot targets this reduces to -log(predicted[true_class]).
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
    raise NotImplementedError(
        "Your turn!\n"
        "  For each (p, t) pair: loss -= t * math.log(max(p, 1e-15))\n"
        "  Return the total loss."
    )


def mse_loss(predicted, target):
    """Compute the Mean Squared Error between predicted and target values.

    Formula:  MSE = (1/n) * sum((predicted_i - target_i)^2)

    Parameters
    ----------
    predicted : list[float] — model's output values
    target    : list[float] — ground-truth values (same length)

    Returns
    -------
    float — non-negative scalar; lower is better
    """
    raise NotImplementedError("Your turn!")


# ---------------------------------------------------------------------------
# Demo (run after implementing the functions above)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing your implementations...")

    # Softmax: outputs should sum to 1
    probs = softmax([2.0, 1.0, 0.5])
    assert abs(sum(probs) - 1.0) < 1e-9, f"softmax sum should be 1.0, got {sum(probs)}"
    assert all(p > 0 for p in probs), "all softmax outputs should be positive"
    print("softmax: PASS")

    # Numerical stability
    big_probs = softmax([1000.0, 1001.0, 1002.0])
    assert abs(sum(big_probs) - 1.0) < 1e-9, "softmax should handle large inputs"
    print("softmax (large inputs, numerical stability): PASS")

    # Cross-entropy: good prediction should have low loss
    target = [0.0, 0.0, 1.0]
    loss_good = cross_entropy_loss([0.01, 0.01, 0.98], target)
    loss_bad = cross_entropy_loss([0.98, 0.01, 0.01], target)
    assert loss_good < 0.1, f"Good prediction should have low loss, got {loss_good}"
    assert loss_bad > 3.0, f"Bad prediction should have high loss, got {loss_bad}"
    print("cross_entropy_loss: PASS")

    # MSE: perfect prediction = 0
    assert mse_loss([1.0, 0.0], [1.0, 0.0]) == 0.0, "MSE of identical vectors should be 0"
    print("mse_loss: PASS")

    print("\nAll tests passed!")
