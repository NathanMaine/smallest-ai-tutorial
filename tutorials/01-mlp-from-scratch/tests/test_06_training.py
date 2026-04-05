"""Tests for Chapter 6: Training Loop (06_training_loop.py)"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
training_mod = importlib.import_module('06_training_loop')
backprop_mod = importlib.import_module('05_backpropagation')
loss_mod = importlib.import_module('04_loss_function')

Trainer = training_mod.Trainer
BackpropNetwork = backprop_mod.BackpropNetwork
softmax = loss_mod.softmax


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_trainer(layer_sizes=(2, 4, 2), lr=0.05, seed=42):
    """Build a small BackpropNetwork wrapped in a Trainer."""
    network = BackpropNetwork(layer_sizes, hidden_activation="relu",
                              output_activation="sigmoid", seed=seed)
    return Trainer(network, learning_rate=lr)


def _predict_class(network, inputs):
    """Return the argmax class index from the network's softmax output."""
    output = network.forward(inputs)
    probs = softmax(output)
    return probs.index(max(probs))


# ---------------------------------------------------------------------------
# test_trainer_creation
# ---------------------------------------------------------------------------

def test_trainer_creation():
    """Trainer stores learning_rate correctly and initialises an empty history."""
    trainer = _make_trainer(lr=0.03)
    assert trainer.learning_rate == 0.03, (
        f"Expected learning_rate=0.03, got {trainer.learning_rate}"
    )
    assert trainer.loss_history == [], (
        f"Expected empty loss_history, got {trainer.loss_history}"
    )
    assert trainer.network is not None, "Trainer.network should not be None"


# ---------------------------------------------------------------------------
# test_single_step_reduces_loss
# ---------------------------------------------------------------------------

def test_single_step_reduces_loss():
    """After 3 training steps on the same input, loss should trend downward.

    We train the *same* (input, target) pair three times.  Because the
    network is adjusted each step to reduce the loss on exactly that pair,
    loss[2] must be strictly less than loss[0].
    """
    trainer = _make_trainer(layer_sizes=[2, 8, 2], lr=0.1, seed=7)
    inputs = [1.0, 0.0]
    target = [1, 0]

    loss1 = trainer.train_step(inputs, target)
    trainer.train_step(inputs, target)
    loss3 = trainer.train_step(inputs, target)

    assert loss3 < loss1, (
        f"Expected loss to decrease after 3 steps on the same example. "
        f"loss1={loss1:.6f}, loss3={loss3:.6f}"
    )


# ---------------------------------------------------------------------------
# test_training_converges_xor_like
# ---------------------------------------------------------------------------

def test_training_converges_xor_like():
    """After 200 epochs, the network correctly classifies all 4 examples.

    Dataset mirrors the task spec: two classes separated by which feature
    dominates.  A [2, 8, 2] network with lr=0.05 is sufficient to solve this.
    """
    data = [
        ([1.0, 0.0], [1, 0]),
        ([0.8, 0.2], [1, 0]),
        ([0.0, 1.0], [0, 1]),
        ([0.2, 0.8], [0, 1]),
    ]

    network = BackpropNetwork([2, 8, 2], hidden_activation="relu",
                              output_activation="sigmoid", seed=42)
    trainer = Trainer(network, learning_rate=0.05)
    trainer.train(data, epochs=200, verbose=False)

    correct = 0
    for inputs, target in data:
        pred = _predict_class(network, inputs)
        true = target.index(max(target))
        correct += int(pred == true)

    assert correct == len(data), (
        f"Expected all {len(data)} examples correct after training, "
        f"got {correct}/{len(data)}."
    )


# ---------------------------------------------------------------------------
# test_training_history_recorded
# ---------------------------------------------------------------------------

def test_training_history_recorded():
    """loss_history must contain exactly one entry per train_step() call."""
    trainer = _make_trainer(lr=0.01, seed=99)
    inputs = [0.5, 0.5]
    target = [1, 0]

    for _ in range(5):
        trainer.train_step(inputs, target)

    assert len(trainer.loss_history) == 5, (
        f"Expected 5 entries in loss_history after 5 steps, "
        f"got {len(trainer.loss_history)}"
    )
    # All recorded values should be finite positive floats
    for i, loss in enumerate(trainer.loss_history):
        assert loss > 0, f"loss_history[{i}] should be positive, got {loss}"
        assert loss < float('inf'), f"loss_history[{i}] should be finite, got {loss}"
