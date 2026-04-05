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
  and outputs from the most recent forward pass. Chapter 6 (backprop) will
  read these to compute gradients without re-running the network.

Chapter roadmap
---------------
  Chapter 1:  Math foundations — vectors & matrices
  Chapter 2:  Single neuron — dot product + bias + activation
  Chapter 3:  Forward pass — Layer and Network classes  ← you are here
  Chapter 4:  Loss functions — MSE, cross-entropy
  Chapter 5:  Backpropagation — gradients via the chain rule
  Chapter 6:  Training loop — putting it all together
"""

import importlib
import random
import sys
import os

# ---------------------------------------------------------------------------
# Import from previous chapters
# ---------------------------------------------------------------------------
# Chapters live in the same directory; add it to sys.path so importlib works
# regardless of where pytest or python3 is invoked from.
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
    weight vector and bias. This is equivalent to the matrix equation:

        z = W @ x + b
        a = activation(z)

    where W has shape (output_size, input_size) — one row per neuron.

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
        Draws each weight independently from Uniform[-limit, limit].

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

        Stores self.inputs, self.z_values, and self.outputs for backprop.

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
            else:  # default: relu
                a = relu(z)

            self.outputs.append(a)

        return self.outputs


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
        num_layers = len(layer_sizes) - 1  # number of weight layers

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

        Parameters
        ----------
        inputs : list[float] — raw input vector for the first layer

        Returns
        -------
        list[float] — output of the final layer
        """
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 phase1-from-scratch/level-a-abcs/03_forward_pass.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3 — Forward Pass Demo")
    print("Layer = many neurons in parallel")
    print("Network = layers stacked in sequence")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Demo 1: Single layer
    # -----------------------------------------------------------------------
    print("\n--- Single Layer (3 inputs → 2 neurons, ReLU) ---")
    layer = Layer(input_size=3, output_size=2, activation="relu", seed=0)
    inputs = [1.0, 2.0, 3.0]
    outputs = layer.forward(inputs)
    print(f"inputs:   {inputs}")
    print(f"weights:  {[[round(w, 4) for w in row] for row in layer.weights]}")
    print(f"biases:   {layer.biases}")
    print(f"z values: {[round(z, 4) for z in layer.z_values]}")
    print(f"outputs:  {[round(a, 4) for a in outputs]}")
    print(f"(ReLU ensures no negative outputs — zeros mean clipped)")

    # -----------------------------------------------------------------------
    # Demo 2: Multi-layer network
    # -----------------------------------------------------------------------
    print("\n--- Multi-layer Network ([4, 8, 4, 2]) ---")
    net = Network([4, 8, 4, 2], hidden_activation="relu",
                  output_activation="sigmoid", seed=42)
    print(f"Architecture: 4 inputs → 8 → 4 → 2 outputs (sigmoid)")
    print(f"Number of layers: {len(net.layers)}")

    net_inputs = [0.5, -0.3, 1.2, 0.8]
    net_outputs = net.forward(net_inputs)
    print(f"\nInput:  {net_inputs}")
    print(f"Output: {[round(v, 6) for v in net_outputs]}")
    print(f"(Sigmoid output layer → values in (0, 1), useful for probabilities)")

    # Inspect stored activations across all layers
    print("\nStored activations per layer after forward pass:")
    for i, lyr in enumerate(net.layers):
        print(f"  Layer {i}: {len(lyr.outputs)} outputs — "
              f"{[round(v, 4) for v in lyr.outputs]}")

    # -----------------------------------------------------------------------
    # Demo 3: One-hot 'a' — 26 inputs representing a letter
    # -----------------------------------------------------------------------
    print("\n--- Forward pass with one-hot letter 'a' (26 inputs) ---")
    # 'a' is index 0 in the alphabet → [1, 0, 0, ..., 0]
    one_hot_a = [1.0] + [0.0] * 25
    print(f"Input (one-hot 'a'): [1.0, 0.0, 0.0, ..., 0.0]  (length {len(one_hot_a)})")

    letter_net = Network([26, 64, 32, 26], hidden_activation="relu",
                         output_activation="sigmoid", seed=7)
    letter_out = letter_net.forward(one_hot_a)
    print(f"Architecture: 26 → 64 → 32 → 26 (sigmoid)")
    print(f"Output length: {len(letter_out)}")
    print(f"Output (first 5): {[round(v, 6) for v in letter_out[:5]]} ...")
    print(f"Min: {round(min(letter_out), 6)}  Max: {round(max(letter_out), 6)}")
    print("(Each output could represent the probability of the next letter)")

    print("\n" + "=" * 60)
    print("Chapter 3 complete. Chapter 4: Loss functions.")
    print("=" * 60)
