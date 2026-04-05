"""Tests for Chapter 2: Single Neuron (02_single_neuron.py)"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
neuron_mod = importlib.import_module('02_single_neuron')
sigmoid = neuron_mod.sigmoid
sigmoid_derivative = neuron_mod.sigmoid_derivative
relu = neuron_mod.relu
relu_derivative = neuron_mod.relu_derivative
Neuron = neuron_mod.Neuron


def test_sigmoid_zero():
    assert sigmoid(0) == 0.5


def test_sigmoid_large_positive():
    assert sigmoid(10) > 0.999


def test_sigmoid_large_negative():
    assert sigmoid(-10) < 0.001


def test_sigmoid_range():
    for x in [-5, -1, 0, 1, 5]:
        assert 0 < sigmoid(x) < 1


def test_sigmoid_derivative_at_half():
    assert sigmoid_derivative(0.5) == 0.25


def test_relu_positive():
    assert relu(5) == 5


def test_relu_negative():
    assert relu(-5) == 0


def test_relu_zero():
    assert relu(0) == 0


def test_relu_derivative_positive():
    assert relu_derivative(5) == 1


def test_relu_derivative_negative():
    assert relu_derivative(-5) == 0


def test_relu_derivative_zero():
    assert relu_derivative(0) == 0


def test_neuron_forward():
    n = Neuron([1, 2, 3], bias=0, activation="relu")
    assert n.forward([1, 1, 1]) == 6


def test_neuron_forward_with_bias():
    n = Neuron([1, 1], bias=-3, activation="relu")
    assert n.forward([1, 1]) == 0  # dot([1,1],[1,1]) + (-3) = -1, relu clips to 0


def test_neuron_sigmoid():
    n = Neuron([0], bias=0, activation="sigmoid")
    assert n.forward([1]) == 0.5  # dot([0],[1]) + 0 = 0, sigmoid(0) = 0.5


def test_neuron_stores_pre_activation():
    n = Neuron([1, 2], bias=1, activation="relu")
    n.forward([3, 4])
    # z = 1*3 + 2*4 + 1 = 3 + 8 + 1 = 12
    assert n.z == 12
    assert n.a == 12  # relu(12) = 12
