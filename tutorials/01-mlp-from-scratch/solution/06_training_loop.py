"""
Chapter 6 — Training Loop
==========================

SGD, the training loop, this is where the magic happens.

We now have every ingredient:

  Chapter 1  — dot products and matrix ops (the atoms)
  Chapter 2  — single neuron: dot + bias + activation
  Chapter 3  — forward pass through multi-layer networks
  Chapter 4  — loss functions: softmax, cross-entropy, MSE
  Chapter 5  — backpropagation: gradients via the chain rule
  Chapter 6  — the training loop: forward → loss → backward → update  ← you are here

The training loop is what turns an untrained network into one that solves
problems. It repeats three fundamental steps:

  1. Forward pass   — predict an output for a given input
  2. Loss           — measure how wrong the prediction is
  3. Backward pass  — compute how each weight contributed to the error
  4. SGD update     — nudge every weight a tiny step against the gradient

Stochastic Gradient Descent (SGD)
----------------------------------
  The "stochastic" part means we update after every single example
  (or a small batch), rather than averaging over the whole dataset.
  This is noisier but much faster to converge and easier to escape
  local minima.

  The update rule for each parameter θ is:

      θ ← θ - learning_rate * dL/dθ

  Subtracting the gradient moves θ in the direction that *decreases* the loss.
  The learning_rate (often called η) controls the step size. Too large and we
  overshoot; too small and training crawls.

Epochs and convergence
-----------------------
  One *epoch* is a full pass through the training dataset.
  We typically train for many epochs, watching the average loss fall.
  When loss stops decreasing, training has *converged*.

What this chapter adds
-----------------------
  Trainer   — wraps a BackpropNetwork with an SGD optimiser.
              Exposes train_step(), train_epoch(), and train().
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

loss_mod = importlib.import_module('04_loss_function')
backprop_mod = importlib.import_module('05_backpropagation')

softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss
BackpropNetwork = backprop_mod.BackpropNetwork


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
        """Store the network and hyperparameters.

        Parameters
        ----------
        network       : BackpropNetwork — the network to train
        learning_rate : float           — SGD step size (default 0.01)
        """
        self.network = network
        self.learning_rate = learning_rate
        self.loss_history = []

    def train_step(self, inputs, target):
        """Perform one forward → loss → backward → SGD update cycle.

        Steps
        -----
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
        # 1. Forward pass
        output = self.network.forward(inputs)

        # 2. Compute loss
        probs = softmax(output)
        loss = cross_entropy_loss(probs, target)

        # 3. Record loss
        self.loss_history.append(loss)

        # 4. Backward pass
        self.network.backward(target)

        # 5. SGD update — nudge every weight and bias against its gradient
        lr = self.learning_rate
        for layer in self.network.layers:
            # Update weights
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] -= lr * layer.weight_gradients[i][j]
            # Update biases
            for i in range(len(layer.biases)):
                layer.biases[i] -= lr * layer.bias_gradients[i]

        return loss

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
        total_loss = 0.0
        for inputs, target in dataset:
            total_loss += self.train_step(inputs, target)
        return total_loss / len(dataset)

    def train(self, dataset, epochs, verbose=True):
        """Train for a fixed number of epochs, printing progress.

        Prints the average loss every 10% of epochs (or every epoch when
        epochs < 10).

        Parameters
        ----------
        dataset : list[tuple[list[float], list[float]]]
            List of (inputs, target) pairs.
        epochs  : int  — number of full passes through the dataset
        verbose : bool — whether to print progress (default True)

        Returns
        -------
        list[float] — average loss for each epoch
        """
        epoch_losses = []
        print_every = max(1, epochs // 10)

        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataset)
            epoch_losses.append(avg_loss)

            if verbose and (epoch == 0 or (epoch + 1) % print_every == 0):
                print(f"  Epoch {epoch + 1:>5}/{epochs}  loss={avg_loss:.6f}")

        return epoch_losses


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 phase1-from-scratch/level-a-abcs/06_training_loop.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("Chapter 6 — Training Loop Demo")
    print("SGD: forward → loss → backward → update")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Helper: predict class from network output
    # -----------------------------------------------------------------------
    def predict_class(network, inputs):
        output = network.forward(inputs)
        probs = softmax(output)
        return probs.index(max(probs))

    # -----------------------------------------------------------------------
    # Dataset: simple 2-class separation problem
    #   Class 0 — examples where first feature > second
    #   Class 1 — examples where second feature > first
    # -----------------------------------------------------------------------
    dataset = [
        ([1.0, 0.0], [1, 0]),
        ([0.9, 0.1], [1, 0]),
        ([0.8, 0.2], [1, 0]),
        ([0.7, 0.3], [1, 0]),
        ([0.0, 1.0], [0, 1]),
        ([0.1, 0.9], [0, 1]),
        ([0.2, 0.8], [0, 1]),
        ([0.3, 0.7], [0, 1]),
    ]

    # -----------------------------------------------------------------------
    # Build network and trainer
    # -----------------------------------------------------------------------
    network = BackpropNetwork([2, 8, 2], hidden_activation="relu",
                              output_activation="sigmoid", seed=42)
    trainer = Trainer(network, learning_rate=0.05)

    # -----------------------------------------------------------------------
    # Predictions BEFORE training
    # -----------------------------------------------------------------------
    print("\n--- Predictions BEFORE training ---")
    correct_before = 0
    for inputs, target in dataset:
        pred = predict_class(network, inputs)
        true = target.index(max(target))
        mark = "OK" if pred == true else "X"
        print(f"  input={[round(x, 1) for x in inputs]}  "
              f"true={true}  pred={pred}  [{mark}]")
        correct_before += int(pred == true)
    accuracy_before = correct_before / len(dataset) * 100
    print(f"\n  Accuracy before: {correct_before}/{len(dataset)} = {accuracy_before:.1f}%")

    # -----------------------------------------------------------------------
    # Train for 200 epochs
    # -----------------------------------------------------------------------
    print("\n--- Training for 200 epochs (lr=0.05) ---")
    epoch_losses = trainer.train(dataset, epochs=200, verbose=True)

    # -----------------------------------------------------------------------
    # Predictions AFTER training
    # -----------------------------------------------------------------------
    print("\n--- Predictions AFTER training ---")
    correct_after = 0
    for inputs, target in dataset:
        pred = predict_class(network, inputs)
        true = target.index(max(target))
        output = network.forward(inputs)
        probs = softmax(output)
        mark = "OK" if pred == true else "X"
        print(f"  input={[round(x, 1) for x in inputs]}  "
              f"true={true}  pred={pred}  "
              f"conf={max(probs):.3f}  [{mark}]")
        correct_after += int(pred == true)
    accuracy_after = correct_after / len(dataset) * 100
    print(f"\n  Accuracy after:  {correct_after}/{len(dataset)} = {accuracy_after:.1f}%")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n  First loss:  {epoch_losses[0]:.6f}")
    print(f"  Final loss:  {epoch_losses[-1]:.6f}")
    print(f"  Improvement: {(1 - epoch_losses[-1] / epoch_losses[0]) * 100:.1f}%")
    print(f"  Steps recorded in loss_history: {len(trainer.loss_history)}")

    print("\n" + "=" * 60)
    print("Chapter 6 complete. The training loop is alive.")
    print("=" * 60)
