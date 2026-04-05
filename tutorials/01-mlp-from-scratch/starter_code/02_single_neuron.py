"""
Chapter 2 — Single Neuron
==========================

A neuron is the fundamental unit of a neural network. It does two things:
1. Computes a weighted sum of its inputs plus a bias: z = dot(weights, inputs) + bias
2. Applies an activation function to introduce nonlinearity: a = activation(z)

Without activation functions, stacking neurons would only produce linear transformations —
no matter how many layers you add, the network could only learn linear boundaries.
Activation functions break linearity, enabling networks to learn complex patterns.

Builds on: 01_math_foundations.py (dot_product)
"""

import importlib
import math
import os
import sys

# Add this file's directory to sys.path so importlib can find sibling modules
_chapter_dir = os.path.dirname(os.path.abspath(__file__))
if _chapter_dir not in sys.path:
    sys.path.insert(0, _chapter_dir)

math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Squishes any value to the range (0, 1). Useful for probabilities.

    Maps large positives near 1, large negatives near 0, and 0 to exactly 0.5.
    Clamps z to [-500, 500] to prevent overflow in exp().

    Formula: 1 / (1 + exp(-z))

    Parameters
    ----------
    z : float — pre-activation value

    Returns
    -------
    float — value in (0, 1)
    """
    raise NotImplementedError("Your turn! Formula: 1 / (1 + math.exp(-z)). Clamp z first.")


def sigmoid_derivative(a):
    """Derivative of sigmoid with respect to z, expressed in terms of the sigmoid OUTPUT a.

    Used in backpropagation to compute gradients.
    Formula: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)

    Parameters
    ----------
    a : float — the output of sigmoid (not the pre-activation z)

    Returns
    -------
    float — gradient of sigmoid at this point
    """
    raise NotImplementedError("Your turn! Formula: a * (1 - a)")


def relu(z):
    """Rectified Linear Unit — returns z if z > 0, else 0.

    Creates sparsity: many neurons output exactly 0, avoiding vanishing gradients.

    Parameters
    ----------
    z : float — pre-activation value

    Returns
    -------
    float — max(0, z)
    """
    raise NotImplementedError("Your turn! Formula: max(0, z)")


def relu_derivative(z):
    """Derivative of ReLU with respect to z (pre-activation value).

    Returns 1 when z > 0 (gradient passes through),
    returns 0 when z <= 0 (gradient blocked — 'dead neuron').

    Parameters
    ----------
    z : float — the pre-activation value (not the relu output)

    Returns
    -------
    int — 1 if z > 0 else 0
    """
    raise NotImplementedError("Your turn!")


# ---------------------------------------------------------------------------
# Neuron class
# ---------------------------------------------------------------------------

class Neuron:
    """A single artificial neuron.

    Computes z = dot(weights, inputs) + bias, then a = activation(z).
    Stores intermediate values (z, a, inputs) for use in backpropagation.
    """

    def __init__(self, weights, bias, activation="relu"):
        """
        Parameters
        ----------
        weights    : list of floats, one per input
        bias       : scalar float added after the weighted sum
        activation : 'relu' or 'sigmoid' (default: 'relu')
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation
        # These are stored after the forward pass for use in backprop
        self.z = 0       # pre-activation value
        self.a = 0       # post-activation output
        self.inputs = [] # inputs seen during this forward pass

    def forward(self, inputs):
        """Compute the neuron's output for the given inputs.

        Steps:
          1. Compute z = dot(weights, inputs) + bias
          2. Apply the activation function to get a
          3. Store inputs, z, a for backprop

        Parameters
        ----------
        inputs : list of floats, same length as self.weights

        Returns
        -------
        float — the activated output a
        """
        raise NotImplementedError(
            "Your turn! "
            "Store inputs, compute z with dot_product, apply activation to get a, "
            "store and return a."
        )


# ---------------------------------------------------------------------------
# Demo (run after implementing everything above)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing your implementations...")

    # Test sigmoid
    s = sigmoid(0)
    assert abs(s - 0.5) < 1e-9, f"sigmoid(0) should be 0.5, got {s}"
    print("sigmoid(0) = 0.5: PASS")

    assert sigmoid(100) > 0.999, "sigmoid(100) should be close to 1"
    print("sigmoid(100) ≈ 1: PASS")

    # Test relu
    assert relu(3) == 3, f"relu(3) should be 3, got {relu(3)}"
    assert relu(-3) == 0, f"relu(-3) should be 0, got {relu(-3)}"
    print("relu: PASS")

    # Test Neuron forward
    n = Neuron(weights=[0.5, -1.0, 2.0], bias=0.1, activation="relu")
    result = n.forward([1.0, 2.0, 3.0])
    # z = 0.5*1 + (-1)*2 + 2*3 + 0.1 = 0.5 - 2 + 6 + 0.1 = 4.6
    assert abs(n.z - 4.6) < 1e-9, f"z should be 4.6, got {n.z}"
    assert result == 4.6, f"relu(4.6) should be 4.6, got {result}"
    print("Neuron forward: PASS")

    print("\nAll tests passed!")
