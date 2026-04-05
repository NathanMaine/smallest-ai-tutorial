"""
Chapter 3 — Forward Pass
========================

Data flows through the network.

  Layer   = many neurons working in parallel on the same input vector.
  Network = layers stacked in sequence — the output of each becomes the
            input of the next.

This chapter wires together the building blocks from Chapters 1 and 2:

  - Chapter 1 supplied dot_product — the weighted sum inside each neuron.
  - Chapter 2 supplied sigmoid and relu — the nonlinear activations.
  - Chapter 3 adds the *structure* that groups neurons into layers and
    chains layers into a complete feed-forward network.

Key ideas
---------
  Xavier (Glorot) initialisation — weights are drawn from a uniform
  distribution in [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
  This keeps activations from exploding or vanishing as depth grows.

  Stored intermediates — every Layer saves its inputs, pre-activations (z),
  and outputs from the most recent forward pass. Chapter 5 (backprop) will
  read these to compute gradients without re-running the network.
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

math_fn = importlib.import_module('01_math_foundations')
neuron_mod = importlib.import_module('02_single_neuron')

dot_product = math_fn.dot_product
sigmoid = neuron_mod.sigmoid
relu = neuron_mod.relu


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class Layer:
    """A layer of neurons that process inputs in parallel.

    Every neuron in the layer sees the same input vector but has its own
    weight vector and bias. Mathematically:

        z = W @ x + b        (W: output_size x input_size)
        a = activation(z)    (applied element-wise)

    Attributes
    ----------
    weights    : list[list[float]] — shape (output_size, input_size)
    biases     : list[float]       — length output_size, initialised to 0
    activation : str               — 'relu' or 'sigmoid'
    inputs     : list[float]       — last input seen (for backprop)
    z_values   : list[float]       — last pre-activation values (for backprop)
    outputs    : list[float]       — last post-activation values (for backprop)
    """

    def __init__(self, input_size, output_size, activation="relu", seed=None):
        """Initialise weights with Xavier uniform, biases at zero.

        Xavier limit: sqrt(6 / (fan_in + fan_out))

        Parameters
        ----------
        input_size  : int  — number of inputs each neuron receives
        output_size : int  — number of neurons (= number of outputs)
        activation  : str  — 'relu' (default) or 'sigmoid'
        seed        : int or None — RNG seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        limit = (6.0 / (input_size + output_size)) ** 0.5

        # weights[i] is the weight vector for neuron i
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.biases = [0.0] * output_size
        self.activation = activation

        # Placeholders — filled in during forward pass; used by backprop
        self.inputs = []
        self.z_values = []
        self.outputs = []

    def forward(self, inputs):
        """Compute the layer's output for the given inputs.

        For each neuron i:
            z[i]      = dot(weights[i], inputs) + biases[i]
            output[i] = activation(z[i])

        Must store self.inputs, self.z_values, and self.outputs for backprop.

        Parameters
        ----------
        inputs : list[float] — length must equal input_size

        Returns
        -------
        list[float] — length output_size, post-activation values
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  1. Store inputs as self.inputs\n"
            "  2. For each neuron i: compute z = dot(weights[i], inputs) + biases[i]\n"
            "  3. Apply self.activation to z\n"
            "  4. Store z_values and outputs\n"
            "  5. Return self.outputs"
        )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """Multi-layer feed-forward network — a stack of Layer objects.

    Data flows from left to right: the output of each Layer becomes the
    input of the next. The final Layer's output is the network's prediction.

    Attributes
    ----------
    layers : list[Layer] — one per adjacent pair in layer_sizes
    """

    def __init__(self, layer_sizes, hidden_activation="relu",
                 output_activation="sigmoid", seed=42):
        """Build a network from a list of layer widths.

        Example: Network([3, 4, 2]) creates:
          - Layer(3 → 4)  with hidden_activation
          - Layer(4 → 2)  with output_activation

        Parameters
        ----------
        layer_sizes        : list[int] — widths of each layer including input
        hidden_activation  : str       — activation for all but the last layer
        output_activation  : str       — activation for the final layer
        seed               : int       — base seed; each layer gets seed+i
        """
        self.layers = []
        num_layers = len(layer_sizes) - 1

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            activation = output_activation if is_last else hidden_activation
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activation,
                seed=seed + i,
            )
            self.layers.append(layer)

    def forward(self, inputs):
        """Run a forward pass through every layer in order.

        The output of each layer becomes the input to the next.

        Parameters
        ----------
        inputs : list[float] — raw input vector for the first layer

        Returns
        -------
        list[float] — output of the final layer
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Hint: start with 'current = inputs', then loop through self.layers."
        )


# ---------------------------------------------------------------------------
# Demo (run after implementing Layer.forward and Network.forward)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing your implementations...")

    # Test Layer
    layer = Layer(input_size=3, output_size=2, activation="relu", seed=0)
    out = layer.forward([1.0, 2.0, 3.0])
    assert len(out) == 2, f"Layer output should have 2 elements, got {len(out)}"
    assert len(layer.inputs) == 3, "Layer should store inputs"
    assert len(layer.z_values) == 2, "Layer should store z_values"
    print("Layer.forward: PASS")

    # Test Network
    net = Network([4, 8, 4, 2])
    out = net.forward([0.5, -0.3, 1.2, 0.8])
    assert len(out) == 2, f"Network output should have 2 elements, got {len(out)}"
    print("Network.forward: PASS")

    print("\nAll tests passed!")
