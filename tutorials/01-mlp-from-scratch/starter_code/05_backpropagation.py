"""
Chapter 5 — Backpropagation
============================

The chain rule, backward pass, how gradients flow.

Everything we have built so far flows data *forward* through the network.
But learning requires flowing *error* backward: from the loss, through each
layer, all the way to the first weight. This is backpropagation.

The core idea is the chain rule from calculus:
    If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

Key equations for a single layer
---------------------------------
Given dL/da (gradient of loss w.r.t. this layer's activations):

  1. dL/dz[i]    = dL/da[i] * activation'(z[i])   — through activation
  2. dL/dW[i][j] = dL/dz[i] * input[j]             — weight gradient
  3. dL/db[i]    = dL/dz[i]                         — bias gradient
  4. dL/dx[j]    = sum_i( dL/dz[i] * W[i][j] )     — for previous layer

Special case: softmax + cross-entropy output gradient
------------------------------------------------------
When the output uses softmax + cross-entropy, the combined gradient at the
output layer simplifies to:
    dL/dz_output = softmax(output) - target

This is why softmax and cross-entropy are almost always paired together.
"""

import importlib
import random
import sys
import os

# ---------------------------------------------------------------------------
# Import from previous chapters
# ---------------------------------------------------------------------------
_chapter_dir = os.path.dirname(os.path.abspath(__file__))
if _chapter_dir not in sys.path:
    sys.path.insert(0, _chapter_dir)

math_fn    = importlib.import_module('01_math_foundations')
neuron_mod = importlib.import_module('02_single_neuron')
loss_mod   = importlib.import_module('04_loss_function')

dot_product        = math_fn.dot_product
sigmoid            = neuron_mod.sigmoid
sigmoid_derivative = neuron_mod.sigmoid_derivative
relu               = neuron_mod.relu
relu_derivative    = neuron_mod.relu_derivative
softmax            = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss


# ---------------------------------------------------------------------------
# BackpropLayer
# ---------------------------------------------------------------------------

class BackpropLayer:
    """A layer with both forward and backward passes.

    Like Layer from Chapter 3, but adds the backward() method that computes
    gradients of the loss with respect to weights, biases, and inputs.

    Attributes
    ----------
    weights          : list[list[float]] — shape (output_size, input_size)
    biases           : list[float]       — length output_size
    activation       : str               — 'relu', 'sigmoid', or 'linear'
    inputs           : list[float]       — cached from last forward()
    z_values         : list[float]       — cached pre-activation values
    outputs          : list[float]       — cached post-activation values
    weight_gradients : list[list[float]] — dL/dW after backward()
    bias_gradients   : list[float]       — dL/db after backward()
    """

    def __init__(self, input_size, output_size, activation="relu", seed=None):
        if seed is not None:
            random.seed(seed)
        limit = (6.0 / (input_size + output_size)) ** 0.5
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.biases = [0.0] * output_size
        self.activation = activation
        self.inputs = []
        self.z_values = []
        self.outputs = []
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, inputs):
        """Compute the layer's output and cache intermediates for backward().

        (Same as Chapter 3's Layer.forward — copy your implementation here
        if you want, or implement fresh.)

        Parameters
        ----------
        inputs : list[float] — length must equal input_size

        Returns
        -------
        list[float] — length output_size, post-activation values
        """
        raise NotImplementedError("Your turn! (Same as Chapter 3's Layer.forward)")

    def backward(self, output_gradients):
        """Compute gradients and propagate error to the previous layer.

        Given dL/da (gradient of loss w.r.t. this layer's activations):
          Step 1: Compute dL/dz via activation derivative
          Step 2: Compute weight gradients: dL/dW[i][j] = dz[i] * input[j]
          Step 3: Compute bias gradients: dL/db[i] = dz[i]
          Step 4: Compute input gradients: dL/dx[j] = sum_i(dz[i] * W[i][j])

        Store weight_gradients and bias_gradients on self.
        Return the input_gradients for the layer before this one.

        Parameters
        ----------
        output_gradients : list[float] — dL/da, length = output_size

        Returns
        -------
        list[float] — input gradients dL/dx, length = input_size
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Step 1: dz[i] = output_gradients[i] * activation_derivative(z or a)\n"
            "          (use sigmoid_derivative(self.outputs[i]) for sigmoid,\n"
            "           relu_derivative(self.z_values[i]) for relu,\n"
            "           output_gradients[i] for linear)\n"
            "  Step 2: self.weight_gradients[i][j] = dz[i] * self.inputs[j]\n"
            "  Step 3: self.bias_gradients[i] = dz[i]\n"
            "  Step 4: input_gradients[j] = sum_i(dz[i] * self.weights[i][j])"
        )


# ---------------------------------------------------------------------------
# BackpropNetwork
# ---------------------------------------------------------------------------

class BackpropNetwork:
    """Multi-layer network with forward and backward passes.

    Attributes
    ----------
    layers : list[BackpropLayer]
    """

    def __init__(self, layer_sizes, hidden_activation="relu",
                 output_activation="sigmoid", seed=42):
        self.layers = []
        self.output_activation = output_activation
        num_layers = len(layer_sizes) - 1
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            activation = output_activation if is_last else hidden_activation
            layer = BackpropLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activation,
                seed=seed + i,
            )
            self.layers.append(layer)

    def forward(self, inputs):
        """Run a forward pass through every layer."""
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, target):
        """Run a backward pass to compute gradients for all layers.

        Uses the softmax + cross-entropy shortcut:
            dL/dz_output = softmax(output) - target

        Then propagates backward through each hidden layer.

        Parameters
        ----------
        target : list[float] — true label (one-hot encoded)
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  1. Get final output: final_output = self.layers[-1].outputs\n"
            "  2. Apply softmax: probs = softmax(final_output)\n"
            "  3. Compute output gradient: grad = [probs[i] - target[i] for i in ...]\n"
            "  4. Loop backward through self.layers, calling layer.backward(grad)\n"
            "     and updating grad each time."
        )


# ---------------------------------------------------------------------------
# Demo (run after implementing backward)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing your implementations...")

    net = BackpropNetwork([2, 3, 2], hidden_activation="sigmoid",
                          output_activation="sigmoid", seed=42)
    inputs = [0.5, -0.3]
    target = [1.0, 0.0]

    output = net.forward(inputs)
    print(f"Forward pass output: {[round(v, 4) for v in output]}")

    net.backward(target)

    for i, layer in enumerate(net.layers):
        assert layer.weight_gradients is not None, f"Layer {i} has no weight gradients"
        assert layer.bias_gradients is not None, f"Layer {i} has no bias gradients"
    print("backward() computes gradients: PASS")

    # Gradient check
    epsilon = 1e-5
    max_diff = 0.0
    for layer_idx, layer in enumerate(net.layers):
        for i in range(len(layer.weights)):
            for j in range(len(layer.weights[i])):
                original = layer.weights[i][j]
                layer.weights[i][j] = original + epsilon
                loss_plus = cross_entropy_loss(softmax(net.forward(inputs)), target)
                layer.weights[i][j] = original - epsilon
                loss_minus = cross_entropy_loss(softmax(net.forward(inputs)), target)
                layer.weights[i][j] = original
                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                net.forward(inputs)
                net.backward(target)
                analytical = layer.weight_gradients[i][j]
                max_diff = max(max_diff, abs(analytical - numerical))

    print(f"Gradient check max diff: {max_diff:.10f}")
    assert max_diff < 1e-4, f"Gradient check FAILED: max_diff={max_diff}"
    print("Gradient check: PASS (analytical ≈ numerical)")
    print("\nAll tests passed!")
