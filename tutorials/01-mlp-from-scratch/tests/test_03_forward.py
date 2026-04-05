"""Tests for Chapter 3: Forward Pass (03_forward_pass.py)"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
forward_mod = importlib.import_module('03_forward_pass')
Layer = forward_mod.Layer
Network = forward_mod.Network


def test_layer_output_size():
    """Layer(3, 2).forward([1,2,3]) produces exactly 2 outputs."""
    layer = Layer(3, 2)
    outputs = layer.forward([1, 2, 3])
    assert len(outputs) == 2


def test_layer_relu_no_negatives():
    """A relu layer never produces negative outputs."""
    layer = Layer(4, 8, activation="relu", seed=0)
    outputs = layer.forward([1.0, 2.0, 3.0, 4.0])
    for val in outputs:
        assert val >= 0, f"ReLU output should not be negative, got {val}"


def test_layer_sigmoid_range():
    """A sigmoid layer produces outputs strictly in (0, 1)."""
    layer = Layer(3, 5, activation="sigmoid", seed=1)
    outputs = layer.forward([0.5, -0.5, 1.0])
    for val in outputs:
        assert 0 < val < 1, f"Sigmoid output should be in (0, 1), got {val}"


def test_layer_deterministic():
    """Same input always produces same output (seed fixed)."""
    layer = Layer(3, 4, activation="relu", seed=42)
    inputs = [1.0, 2.0, 3.0]
    out1 = layer.forward(inputs)
    out2 = layer.forward(inputs)
    assert out1 == out2


def test_network_creation():
    """Network([3, 4, 2]) has exactly 2 layers."""
    net = Network([3, 4, 2])
    assert len(net.layers) == 2


def test_network_forward_output_size():
    """Network([3, 4, 2]).forward([1, 2, 3]) produces 2 outputs."""
    net = Network([3, 4, 2])
    outputs = net.forward([1, 2, 3])
    assert len(outputs) == 2


def test_network_deep_forward():
    """Network([2, 8, 8, 4]) produces 4 outputs, all >= 0 (relu hidden layers)."""
    net = Network([2, 8, 8, 4], hidden_activation="relu", output_activation="sigmoid")
    outputs = net.forward([0.5, -0.5])
    assert len(outputs) == 4
    # output layer uses sigmoid so all values in (0, 1) >= 0
    for val in outputs:
        assert val >= 0


def test_network_stores_all_activations():
    """After forward pass, every layer has non-empty outputs stored."""
    net = Network([3, 5, 4, 2])
    net.forward([1.0, 0.0, -1.0])
    for i, layer in enumerate(net.layers):
        assert len(layer.outputs) > 0, f"Layer {i} has no stored outputs after forward pass"
