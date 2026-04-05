"""
Chapter 6 — Training Loop
==========================

SGD, the training loop, this is where the magic happens.

The training loop is what turns an untrained network into one that solves
problems. It repeats four steps:

  1. Forward pass   — predict an output for a given input
  2. Loss           — measure how wrong the prediction is
  3. Backward pass  — compute how each weight contributed to the error
  4. SGD update     — nudge every weight a tiny step against the gradient

Stochastic Gradient Descent (SGD)
----------------------------------
The update rule for each parameter θ is:
    θ ← θ - learning_rate * dL/dθ

Subtracting the gradient moves θ in the direction that *decreases* the loss.

"Stochastic" = we update after every single example (not after averaging
over the whole dataset). Noisier, but often faster to converge.
"""

import importlib
import os
import sys

# ---------------------------------------------------------------------------
# Import from previous chapters
# ---------------------------------------------------------------------------
_chapter_dir = os.path.dirname(os.path.abspath(__file__))
if _chapter_dir not in sys.path:
    sys.path.insert(0, _chapter_dir)

loss_mod    = importlib.import_module('04_loss_function')
backprop_mod = importlib.import_module('05_backpropagation')

softmax           = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss
BackpropNetwork   = backprop_mod.BackpropNetwork


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Wraps a BackpropNetwork and trains it with vanilla SGD.

    Attributes
    ----------
    network       : BackpropNetwork — the network being trained
    learning_rate : float           — step size for SGD updates
    loss_history  : list[float]     — loss recorded after every train_step()
    """

    def __init__(self, network, learning_rate=0.01):
        self.network = network
        self.learning_rate = learning_rate
        self.loss_history = []

    def train_step(self, inputs, target):
        """Perform one forward → loss → backward → SGD update cycle.

        Steps:
          1. Forward pass  — run inputs through the network
          2. Compute loss  — softmax over output, then cross-entropy
          3. Record loss   — append to loss_history
          4. Backward pass — compute gradients for every weight and bias
          5. SGD update    — subtract learning_rate * gradient from each parameter

        Parameters
        ----------
        inputs : list[float] — input feature vector
        target : list[float] — one-hot encoded true label

        Returns
        -------
        float — the cross-entropy loss before the update
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  1. output = self.network.forward(inputs)\n"
            "  2. probs = softmax(output)\n"
            "  3. loss = cross_entropy_loss(probs, target)\n"
            "  4. self.loss_history.append(loss)\n"
            "  5. self.network.backward(target)\n"
            "  6. For each layer, for each weight: w -= learning_rate * gradient\n"
            "     For each bias: b -= learning_rate * gradient\n"
            "  7. return loss"
        )

    def train_epoch(self, dataset):
        """Train on every example in the dataset once.

        Parameters
        ----------
        dataset : list[tuple[list[float], list[float]]]
            List of (inputs, target) pairs.

        Returns
        -------
        float — average loss across all examples in the epoch
        """
        raise NotImplementedError(
            "Your turn! Call train_step for each (inputs, target), return average loss."
        )

    def train(self, dataset, epochs, verbose=True):
        """Train for a fixed number of epochs, printing progress.

        Parameters
        ----------
        dataset : list[tuple[list[float], list[float]]]
        epochs  : int  — number of full passes through the dataset
        verbose : bool — whether to print progress (default True)

        Returns
        -------
        list[float] — average loss for each epoch
        """
        raise NotImplementedError(
            "Your turn! Call train_epoch for each epoch, collect and return losses."
        )


# ---------------------------------------------------------------------------
# Demo (run after implementing Trainer)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("Testing your implementation...")

    # Simple 2-class dataset
    dataset = [
        ([1.0, 0.0], [1, 0]),
        ([0.9, 0.1], [1, 0]),
        ([0.0, 1.0], [0, 1]),
        ([0.1, 0.9], [0, 1]),
    ]

    network = BackpropNetwork([2, 4, 2], hidden_activation="relu",
                              output_activation="sigmoid", seed=42)
    trainer = Trainer(network, learning_rate=0.1)

    # Train for 100 epochs
    losses = trainer.train(dataset, epochs=100, verbose=False)

    assert len(losses) == 100, f"Expected 100 epoch losses, got {len(losses)}"
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}: PASS")

    # Check predictions
    def predict(net, inp):
        out = net.forward(inp)
        probs = softmax(out)
        return probs.index(max(probs))

    correct = sum(1 for x, t in dataset if predict(network, x) == t.index(max(t)))
    print(f"Accuracy after training: {correct}/{len(dataset)}")
    assert correct == len(dataset), "Should classify all examples correctly after 100 epochs"
    print("Training loop: PASS")
    print("\nAll tests passed!")
