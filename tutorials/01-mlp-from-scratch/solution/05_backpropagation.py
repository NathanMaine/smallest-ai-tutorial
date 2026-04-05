"""
Chapter 5 — Backpropagation
============================

The chain rule, backward pass, how we compute gradients to enable learning.

Everything we have built so far — dot products, activations, layers, loss
functions — flows data *forward* through the network. But learning requires
flowing *error* backward: from the loss, through each layer, all the way
to the first weight. This reverse flow is **backpropagation**.

The core idea is the **chain rule** from calculus:

    If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

In a neural network, the "chain" has many links:

    input → z = Wx + b → a = activation(z) → ... → loss

Backpropagation computes dL/dW (how much each weight contributed to the
error) by multiplying local derivatives along the chain, moving from the
loss back to the input.

Key equations for a single layer
---------------------------------
  Given dL/da (gradient of loss w.r.t. this layer's activations):

  1. dL/dz[i]  = dL/da[i] * da/dz[i]           (activation derivative)
  2. dL/dw[i][j] = dL/dz[i] * input[j]          (weight gradient)
  3. dL/db[i]    = dL/dz[i]                      (bias gradient)
  4. dL/dx[j]    = sum_i( dL/dz[i] * w[i][j] )  (input gradient for prev layer)

Special case: softmax + cross-entropy
--------------------------------------
  When the output layer uses softmax activation with cross-entropy loss,
  the combined gradient simplifies beautifully:

      dL/dz = softmax(z) - target

  This avoids computing the full Jacobian of softmax and is both simpler
  and numerically more stable.

Chapter roadmap
---------------
  Chapter 1:  Math foundations — vectors & matrices
  Chapter 2:  Single neuron — dot product + bias + activation
  Chapter 3:  Forward pass — Layer and Network classes
  Chapter 4:  Loss functions — softmax, cross-entropy, MSE
  Chapter 5:  Backpropagation — gradients via the chain rule  ← you are here
  Chapter 6:  Training loop — putting it all together
"""

import importlib
import math
import random
import sys
import os

# ---------------------------------------------------------------------------
# Import from previous chapters
# ---------------------------------------------------------------------------
_chapter_dir = os.path.dirname(os.path.abspath(__file__))
if _chapter_dir not in sys.path:
    sys.path.insert(0, _chapter_dir)

math_fn = importlib.import_module('01_math_foundations')
neuron_mod = importlib.import_module('02_single_neuron')
loss_mod = importlib.import_module('04_loss_function')

dot_product = math_fn.dot_product
sigmoid = neuron_mod.sigmoid
sigmoid_derivative = neuron_mod.sigmoid_derivative
relu = neuron_mod.relu
relu_derivative = neuron_mod.relu_derivative
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss


# ---------------------------------------------------------------------------
# BackpropLayer
# ---------------------------------------------------------------------------

class BackpropLayer:
    """A layer of neurons with both forward and backward passes.

    Like Layer from Chapter 3, but adds the backward() method that computes
    gradients of the loss with respect to weights, biases, and inputs.

    Attributes
    ----------
    weights         : list[list[float]] — shape (output_size, input_size)
    biases          : list[float]       — length output_size
    activation      : str               — 'relu', 'sigmoid', or 'linear'
    inputs          : list[float]       — cached from last forward()
    z_values        : list[float]       — cached pre-activation values
    outputs         : list[float]       — cached post-activation values
    weight_gradients: list[list[float]] or None — dL/dW after backward()
    bias_gradients  : list[float] or None       — dL/db after backward()
    """

    def __init__(self, input_size, output_size, activation="relu", seed=None):
        """Initialise with Xavier uniform weights and zero biases.

        Parameters
        ----------
        input_size  : int  — number of inputs each neuron receives
        output_size : int  — number of neurons
        activation  : str  — 'relu', 'sigmoid', or 'linear'
        seed        : int or None — RNG seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        limit = (6.0 / (input_size + output_size)) ** 0.5
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.biases = [0.0] * output_size
        self.activation = activation

        # Cached values from forward pass
        self.inputs = []
        self.z_values = []
        self.outputs = []

        # Gradients — populated by backward()
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, inputs):
        """Compute the layer's output and cache intermediates for backward().

        Parameters
        ----------
        inputs : list[float] — length must equal input_size

        Returns
        -------
        list[float] — length output_size, post-activation values
        """
        self.inputs = inputs
        self.z_values = []
        self.outputs = []

        for i in range(len(self.weights)):
            z = dot_product(self.weights[i], inputs) + self.biases[i]
            self.z_values.append(z)

            if self.activation == "sigmoid":
                a = sigmoid(z)
            elif self.activation == "linear":
                a = z
            else:  # relu
                a = relu(z)

            self.outputs.append(a)

        return self.outputs

    def backward(self, output_gradients):
        """Compute gradients and propagate error to the previous layer.

        Given dL/da (gradient of loss w.r.t. this layer's activations),
        computes:
          - dL/dz  via the activation derivative
          - dL/dW  (weight gradients)
          - dL/db  (bias gradients)
          - dL/dx  (input gradients for the previous layer)

        Parameters
        ----------
        output_gradients : list[float] — dL/da, length = output_size

        Returns
        -------
        list[float] — input gradients dL/dx, length = input_size
        """
        output_size = len(self.weights)
        input_size = len(self.weights[0])

        # Step 1: Compute dL/dz (pre-activation gradients)
        dz = [0.0] * output_size
        for i in range(output_size):
            if self.activation == "sigmoid":
                dz[i] = output_gradients[i] * sigmoid_derivative(self.outputs[i])
            elif self.activation == "relu":
                dz[i] = output_gradients[i] * relu_derivative(self.z_values[i])
            else:  # linear
                dz[i] = output_gradients[i]

        # Step 2: Compute weight gradients — dL/dw[i][j] = dz[i] * input[j]
        self.weight_gradients = [
            [dz[i] * self.inputs[j] for j in range(input_size)]
            for i in range(output_size)
        ]

        # Step 3: Compute bias gradients — dL/db[i] = dz[i]
        self.bias_gradients = [dz[i] for i in range(output_size)]

        # Step 4: Compute input gradients — dL/dx[j] = sum_i(dz[i] * w[i][j])
        input_gradients = [0.0] * input_size
        for j in range(input_size):
            for i in range(output_size):
                input_gradients[j] += dz[i] * self.weights[i][j]

        return input_gradients


# ---------------------------------------------------------------------------
# BackpropNetwork
# ---------------------------------------------------------------------------

class BackpropNetwork:
    """Multi-layer network with forward and backward passes.

    The backward pass uses the well-known softmax + cross-entropy
    simplification: the gradient at the output layer is simply
    softmax(output) - target.

    Attributes
    ----------
    layers : list[BackpropLayer]
    """

    def __init__(self, layer_sizes, hidden_activation="relu",
                 output_activation="sigmoid", seed=42):
        """Build a network from a list of layer widths.

        Parameters
        ----------
        layer_sizes        : list[int] — widths including input layer
        hidden_activation  : str       — activation for hidden layers
        output_activation  : str       — activation for the output layer
        seed               : int       — base seed; each layer gets seed+i
        """
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
        """Run a forward pass through every layer.

        Parameters
        ----------
        inputs : list[float] — raw input vector

        Returns
        -------
        list[float] — output of the final layer
        """
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, target):
        """Run a backward pass to compute gradients for all layers.

        Uses the softmax + cross-entropy shortcut for the output gradient:
            dL/dz_output = softmax(output) - target

        Then propagates backward through each hidden layer.

        Parameters
        ----------
        target : list[float] — true label (one-hot encoded)
        """
        # The output layer produced raw activations (sigmoid/linear).
        # For the loss gradient with softmax + cross-entropy, we apply
        # softmax to the final layer's *pre-activation* values (z) and
        # compute the simplified gradient.
        #
        # However, since our network applies sigmoid/relu as the output
        # activation, the clean approach is:
        #   - Treat the final layer output as logits for softmax
        #   - The combined gradient dL/d(output) = softmax(output) - target
        #   - Then backprop through the output layer's activation normally
        #
        # For sigmoid output: this gives correct gradients because
        # the chain rule through sigmoid is handled in layer.backward().

        final_output = self.layers[-1].outputs
        probs = softmax(final_output)

        # dL/da for the output layer = softmax(a) - target
        output_grad = [probs[i] - target[i] for i in range(len(target))]

        # Propagate backward through all layers
        grad = output_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 phase1-from-scratch/level-a-abcs/05_backpropagation.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 5 — Backpropagation Demo")
    print("The chain rule: how gradients flow backward")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Demo 1: Forward + backward on a small network
    # -----------------------------------------------------------------------
    print("\n--- Small Network [2, 3, 2] with sigmoid ---")
    net = BackpropNetwork([2, 3, 2], hidden_activation="sigmoid",
                          output_activation="sigmoid", seed=42)
    inputs = [0.5, -0.3]
    target = [1.0, 0.0]

    output = net.forward(inputs)
    print(f"Input:  {inputs}")
    print(f"Target: {target}")
    print(f"Output: {[round(v, 6) for v in output]}")

    probs = softmax(output)
    loss = cross_entropy_loss(probs, target)
    print(f"Probs:  {[round(p, 6) for p in probs]}")
    print(f"Loss:   {round(loss, 6)}")

    net.backward(target)

    print("\nGradients after backward pass:")
    for i, layer in enumerate(net.layers):
        print(f"\n  Layer {i}:")
        print(f"    Weight gradients shape: {len(layer.weight_gradients)}x{len(layer.weight_gradients[0])}")
        print(f"    Bias gradients shape:   {len(layer.bias_gradients)}")
        print(f"    Weight gradients: {[[round(g, 6) for g in row] for row in layer.weight_gradients]}")
        print(f"    Bias gradients:   {[round(g, 6) for g in layer.bias_gradients]}")

    # -----------------------------------------------------------------------
    # Demo 2: Numerical gradient check
    # -----------------------------------------------------------------------
    print("\n--- Numerical Gradient Check ---")
    print("Verifying analytical gradients match numerical approximation...")

    epsilon = 1e-5
    max_diff = 0.0
    checks = 0

    for layer_idx, layer in enumerate(net.layers):
        for i in range(len(layer.weights)):
            for j in range(len(layer.weights[i])):
                original = layer.weights[i][j]

                # f(w + eps)
                layer.weights[i][j] = original + epsilon
                out_plus = net.forward(inputs)
                loss_plus = cross_entropy_loss(softmax(out_plus), target)

                # f(w - eps)
                layer.weights[i][j] = original - epsilon
                out_minus = net.forward(inputs)
                loss_minus = cross_entropy_loss(softmax(out_minus), target)

                layer.weights[i][j] = original
                numerical = (loss_plus - loss_minus) / (2 * epsilon)

                # Re-run forward/backward to get fresh analytical gradients
                net.forward(inputs)
                net.backward(target)
                analytical = layer.weight_gradients[i][j]

                diff = abs(analytical - numerical)
                max_diff = max(max_diff, diff)
                checks += 1

    print(f"  Checked {checks} weight gradients")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Status: {'PASS' if max_diff < 1e-4 else 'FAIL'} (tolerance: 1e-4)")

    print("\n" + "=" * 60)
    print("Chapter 5 complete. Chapter 6: Training loop.")
    print("=" * 60)
