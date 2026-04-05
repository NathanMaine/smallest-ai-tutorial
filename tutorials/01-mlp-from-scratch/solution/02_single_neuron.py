"""
Chapter 2: Single Neuron

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
    """
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid with respect to z, expressed in terms of the sigmoid OUTPUT a.

    Used in backpropagation to compute gradients.
    Formula: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)

    Args:
        a: the output of sigmoid (not the pre-activation z)
    """
    return a * (1 - a)


def relu(z):
    """Rectified Linear Unit — the most popular modern activation function.

    Returns z if z > 0, else 0. Creates sparsity: many neurons output exactly 0,
    making the network efficient and avoiding the vanishing gradient problem.
    """
    return max(0, z)


def relu_derivative(z):
    """Derivative of ReLU with respect to z (pre-activation value).

    Full pass-through (gradient = 1) when z > 0, completely blocked (gradient = 0) otherwise.

    Args:
        z: the pre-activation value (not the relu output)
    """
    return 1 if z > 0 else 0


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
        Args:
            weights: list of floats, one per input
            bias: scalar float added after the weighted sum
            activation: 'relu' or 'sigmoid' (default: 'relu')
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation
        # Stored after forward pass
        self.z = 0       # pre-activation value
        self.a = 0       # post-activation output
        self.inputs = [] # inputs seen during forward pass

    def forward(self, inputs):
        """Compute the neuron's output for the given inputs.

        Args:
            inputs: list of floats, same length as self.weights

        Returns:
            a: the activated output (float)
        """
        self.inputs = inputs
        self.z = dot_product(self.weights, inputs) + self.bias

        if self.activation == "sigmoid":
            self.a = sigmoid(self.z)
        else:  # default: relu
            self.a = relu(self.z)

        return self.a


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("Chapter 2: Single Neuron")
    print("=" * 50)

    print("\n--- Activation Functions ---")
    test_values = [-10, -2, -1, 0, 1, 2, 10]
    print(f"{'z':>6}  {'sigmoid(z)':>12}  {'relu(z)':>8}")
    print("-" * 32)
    for z in test_values:
        print(f"{z:>6}  {sigmoid(z):>12.6f}  {relu(z):>8}")

    print("\n--- Sigmoid Derivative ---")
    print(f"sigmoid_derivative(0.5) = {sigmoid_derivative(0.5)}")
    print(f"sigmoid_derivative(sigmoid(0)) = {sigmoid_derivative(sigmoid(0))}")

    print("\n--- ReLU Derivative ---")
    for z in [-3, 0, 3]:
        print(f"relu_derivative({z}) = {relu_derivative(z)}")

    print("\n--- Neuron Forward Pass (ReLU) ---")
    n_relu = Neuron(weights=[0.5, -1.0, 2.0], bias=0.1, activation="relu")
    result = n_relu.forward([1.0, 2.0, 3.0])
    print(f"weights={n_relu.weights}, bias={n_relu.bias}")
    print(f"inputs=[1.0, 2.0, 3.0]")
    print(f"z = dot(weights, inputs) + bias = {n_relu.z:.4f}")
    print(f"a = relu(z) = {result:.4f}")

    print("\n--- Neuron Forward Pass (Sigmoid) ---")
    n_sig = Neuron(weights=[1.0, 1.0], bias=0.0, activation="sigmoid")
    result2 = n_sig.forward([0.5, -0.5])
    print(f"weights={n_sig.weights}, bias={n_sig.bias}")
    print(f"inputs=[0.5, -0.5]")
    print(f"z = {n_sig.z:.4f}")
    print(f"a = sigmoid(z) = {result2:.4f}")

    print("\n--- Neuron clipping negative output with ReLU ---")
    n_clip = Neuron(weights=[1, 1], bias=-3, activation="relu")
    out = n_clip.forward([1, 1])
    print(f"dot([1,1],[1,1]) + (-3) = {n_clip.z}  →  relu({n_clip.z}) = {out}")
    print("(negative pre-activation is clipped to 0 by ReLU)")
