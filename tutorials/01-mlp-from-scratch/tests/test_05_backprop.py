"""Tests for Chapter 5: Backpropagation (05_backpropagation.py)"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
bp_mod = importlib.import_module('05_backpropagation')
loss_mod = importlib.import_module('04_loss_function')

BackpropLayer = bp_mod.BackpropLayer
BackpropNetwork = bp_mod.BackpropNetwork
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss


# ---------------------------------------------------------------------------
# Helper: numerical gradient via central differences
# ---------------------------------------------------------------------------

def _numerical_gradient(net, inputs, target, layer_idx, i, j, epsilon=1e-5):
    """Central difference numerical gradient for weight[i][j] in given layer."""
    original = net.layers[layer_idx].weights[i][j]

    net.layers[layer_idx].weights[i][j] = original + epsilon
    out_plus = net.forward(inputs)
    probs_plus = softmax(out_plus)
    loss_plus = cross_entropy_loss(probs_plus, target)

    net.layers[layer_idx].weights[i][j] = original - epsilon
    out_minus = net.forward(inputs)
    probs_minus = softmax(out_minus)
    loss_minus = cross_entropy_loss(probs_minus, target)

    net.layers[layer_idx].weights[i][j] = original
    return (loss_plus - loss_minus) / (2 * epsilon)


def _numerical_bias_gradient(net, inputs, target, layer_idx, i, epsilon=1e-5):
    """Central difference numerical gradient for bias[i] in given layer."""
    original = net.layers[layer_idx].biases[i]

    net.layers[layer_idx].biases[i] = original + epsilon
    out_plus = net.forward(inputs)
    probs_plus = softmax(out_plus)
    loss_plus = cross_entropy_loss(probs_plus, target)

    net.layers[layer_idx].biases[i] = original - epsilon
    out_minus = net.forward(inputs)
    probs_minus = softmax(out_minus)
    loss_minus = cross_entropy_loss(probs_minus, target)

    net.layers[layer_idx].biases[i] = original
    return (loss_plus - loss_minus) / (2 * epsilon)


# ---------------------------------------------------------------------------
# BackpropLayer tests
# ---------------------------------------------------------------------------

def test_backprop_layer_forward_matches_layer():
    """BackpropLayer.forward should produce the same result as Layer."""
    layer = BackpropLayer(3, 2, activation="sigmoid", seed=10)
    inputs = [0.5, -0.3, 0.8]
    outputs = layer.forward(inputs)
    assert len(outputs) == 2
    assert layer.inputs == inputs
    assert len(layer.z_values) == 2
    assert len(layer.outputs) == 2


def test_backprop_produces_gradients():
    """After backward(), every layer has weight_gradients and bias_gradients."""
    net = BackpropNetwork([2, 3, 2], hidden_activation="sigmoid",
                          output_activation="sigmoid", seed=42)
    inputs = [0.5, -0.3]
    target = [1.0, 0.0]

    net.forward(inputs)
    net.backward(target)

    for idx, layer in enumerate(net.layers):
        assert layer.weight_gradients is not None, \
            f"Layer {idx} weight_gradients is None after backward()"
        assert layer.bias_gradients is not None, \
            f"Layer {idx} bias_gradients is None after backward()"


def test_backprop_gradient_shapes():
    """Gradient shapes must match weight shapes."""
    net = BackpropNetwork([2, 3, 2], hidden_activation="sigmoid",
                          output_activation="sigmoid", seed=42)
    inputs = [0.5, -0.3]
    target = [1.0, 0.0]

    net.forward(inputs)
    net.backward(target)

    for idx, layer in enumerate(net.layers):
        # weight_gradients shape should match weights shape
        assert len(layer.weight_gradients) == len(layer.weights), \
            f"Layer {idx}: weight_gradients rows {len(layer.weight_gradients)} != weights rows {len(layer.weights)}"
        for i in range(len(layer.weights)):
            assert len(layer.weight_gradients[i]) == len(layer.weights[i]), \
                f"Layer {idx}, neuron {i}: weight_gradients cols don't match"

        # bias_gradients length should match biases length
        assert len(layer.bias_gradients) == len(layer.biases), \
            f"Layer {idx}: bias_gradients length {len(layer.bias_gradients)} != biases length {len(layer.biases)}"


def test_backprop_numerical_gradient_check():
    """CRITICAL: Analytical gradients must match numerical gradients within 1e-4.

    Uses sigmoid activation (smooth everywhere) to avoid ReLU discontinuity issues.
    Tests several weights in each layer of a [2, 3, 2] network.
    """
    net = BackpropNetwork([2, 3, 2], hidden_activation="sigmoid",
                          output_activation="sigmoid", seed=42)
    inputs = [0.5, -0.3]
    target = [1.0, 0.0]

    # Run forward + backward to get analytical gradients
    net.forward(inputs)
    net.backward(target)

    tolerance = 1e-4

    # Check every weight in every layer
    for layer_idx, layer in enumerate(net.layers):
        for i in range(len(layer.weights)):
            for j in range(len(layer.weights[i])):
                analytical = layer.weight_gradients[i][j]
                numerical = _numerical_gradient(net, inputs, target,
                                                layer_idx, i, j)
                diff = abs(analytical - numerical)
                assert diff < tolerance, (
                    f"Layer {layer_idx}, weight[{i}][{j}]: "
                    f"analytical={analytical:.8f}, numerical={numerical:.8f}, "
                    f"diff={diff:.8f} (tolerance={tolerance})"
                )


def test_backprop_numerical_bias_gradient_check():
    """Bias gradients must also match numerical approximation."""
    net = BackpropNetwork([2, 3, 2], hidden_activation="sigmoid",
                          output_activation="sigmoid", seed=42)
    inputs = [0.5, -0.3]
    target = [1.0, 0.0]

    net.forward(inputs)
    net.backward(target)

    tolerance = 1e-4

    for layer_idx, layer in enumerate(net.layers):
        for i in range(len(layer.biases)):
            analytical = layer.bias_gradients[i]
            numerical = _numerical_bias_gradient(net, inputs, target,
                                                 layer_idx, i)
            diff = abs(analytical - numerical)
            assert diff < tolerance, (
                f"Layer {layer_idx}, bias[{i}]: "
                f"analytical={analytical:.8f}, numerical={numerical:.8f}, "
                f"diff={diff:.8f} (tolerance={tolerance})"
            )


def test_backprop_with_relu_hidden():
    """Backprop works with ReLU hidden layers (basic smoke test)."""
    net = BackpropNetwork([2, 4, 2], hidden_activation="relu",
                          output_activation="sigmoid", seed=7)
    inputs = [1.0, 0.5]
    target = [0.0, 1.0]

    net.forward(inputs)
    net.backward(target)

    # Just check gradients exist and have correct shapes
    for layer in net.layers:
        assert layer.weight_gradients is not None
        assert layer.bias_gradients is not None


def test_backprop_linear_output():
    """BackpropLayer with linear activation passes gradients through unchanged."""
    layer = BackpropLayer(2, 2, activation="linear", seed=5)
    layer.forward([1.0, 0.5])
    input_grads = layer.backward([1.0, -1.0])
    # With linear activation, dz = output_gradients directly
    assert layer.bias_gradients == [1.0, -1.0]
    assert len(input_grads) == 2
