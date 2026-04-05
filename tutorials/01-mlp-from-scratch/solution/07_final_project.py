"""
Chapter 7 — Letter Classifier
==============================

Everything comes together: the first complete working AI.

A neural network that classifies all 26 letters of the alphabet using only
the foundations built in Chapters 1-6 — no external ML libraries, no numpy,
no torch, just pure Python arithmetic and the chain rule.

What this chapter demonstrates
--------------------------------
  - A real classification problem: 26 classes, 26-dimensional input
  - One-hot encoding: how to represent categorical data as vectors
  - The full pipeline: data → train → evaluate → predict
  - Softmax probability distributions over multiple classes
  - 95%+ accuracy from a tiny network (~13.7 KB of parameters)

Architecture
------------
  Input layer   : 26 neurons  (one-hot letter encoding)
  Hidden layer  : 64 neurons  (ReLU activation)
  Output layer  : 26 neurons  (sigmoid activation → softmax for prediction)

  Parameters: 26×64 + 64 + 64×26 + 26 = 1664 + 64 + 1664 + 26 = 3418
  At 4 bytes each (float32 equivalent): ~13.7 KB

Training
--------
  Dataset: 26 examples — for each letter i, input = one-hot[i], target = one-hot[i]
  The network learns to map each letter's one-hot vector back to itself.
  SGD with learning_rate=0.05 converges to >95% accuracy in 500 epochs.

Chapter roadmap
---------------
  Chapter 1:  Math foundations — vectors & matrices
  Chapter 2:  Single neuron — dot product + bias + activation
  Chapter 3:  Forward pass — Layer and Network classes
  Chapter 4:  Loss functions — softmax, cross-entropy, MSE
  Chapter 5:  Backpropagation — gradients via the chain rule
  Chapter 6:  Training loop — putting it all together
  Chapter 7:  Letter classifier — the first working AI  ← you are here
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

backprop_mod = importlib.import_module('05_backpropagation')
trainer_mod = importlib.import_module('06_training_loop')
loss_mod = importlib.import_module('04_loss_function')

BackpropNetwork = backprop_mod.BackpropNetwork
Trainer = trainer_mod.Trainer
softmax = loss_mod.softmax


# ---------------------------------------------------------------------------
# LetterClassifier
# ---------------------------------------------------------------------------

class LetterClassifier:
    """A neural network that recognises all 26 letters of the alphabet.

    Internally it maps each letter's one-hot input vector back to itself,
    learning a 26-class classification function from scratch.

    Attributes
    ----------
    LETTERS        : str               — 'abcdefghijklmnopqrstuvwxyz'
    hidden_size    : int               — number of hidden neurons
    learning_rate  : float             — SGD step size
    network        : BackpropNetwork   — the underlying neural network
    trainer        : Trainer           — the SGD trainer wrapping the network
    """

    LETTERS = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self, hidden_size=64, learning_rate=0.05, seed=42):
        """Build the network and trainer.

        Parameters
        ----------
        hidden_size   : int   — neurons in the hidden layer (default 64)
        learning_rate : float — SGD step size (default 0.05)
        seed          : int   — RNG seed for reproducibility (default 42)
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Architecture: [26 inputs] → [hidden_size ReLU] → [26 sigmoid outputs]
        self.network = BackpropNetwork(
            [26, hidden_size, 26],
            hidden_activation="relu",
            output_activation="sigmoid",
            seed=seed,
        )
        self.trainer = Trainer(self.network, learning_rate=learning_rate)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _make_dataset(self):
        """Create 26 training examples — one per letter.

        For each letter i (0-25):
          input  = one-hot vector with a 1 at position i, 0 everywhere else
          target = same one-hot vector

        The network learns to identify which letter it is given.

        Returns
        -------
        list[tuple[list[float], list[float]]]
            26 (input, target) pairs, each vector of length 26.
        """
        dataset = []
        for i in range(26):
            one_hot = [0.0] * 26
            one_hot[i] = 1.0
            dataset.append((one_hot, one_hot[:]))  # copy target to be safe
        return dataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, epochs=500, verbose=True):
        """Train the classifier on all 26 letters.

        Parameters
        ----------
        epochs  : int  — number of full passes through the 26-letter dataset
        verbose : bool — whether to print loss progress (default True)

        Returns
        -------
        list[float] — average loss per epoch
        """
        dataset = self._make_dataset()
        if verbose:
            print(f"Training letter classifier for {epochs} epochs "
                  f"(lr={self.learning_rate}, hidden={self.hidden_size})...")
        epoch_losses = self.trainer.train(dataset, epochs=epochs, verbose=verbose)
        if verbose:
            print(f"  Final loss: {epoch_losses[-1]:.6f}")
        return epoch_losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, one_hot_input):
        """Predict the letter for a one-hot input vector.

        Parameters
        ----------
        one_hot_input : list[float] — length-26 one-hot vector

        Returns
        -------
        str — the predicted letter (single lowercase character)
        """
        raw_output = self.network.forward(one_hot_input)
        probs = softmax(raw_output)
        best_index = probs.index(max(probs))
        return self.LETTERS[best_index]

    def predict_probs(self, one_hot_input):
        """Return a probability distribution over all 26 letters.

        Parameters
        ----------
        one_hot_input : list[float] — length-26 one-hot vector

        Returns
        -------
        dict[str, float] — mapping from each letter to its probability
        """
        raw_output = self.network.forward(one_hot_input)
        probs = softmax(raw_output)
        return {letter: probs[i] for i, letter in enumerate(self.LETTERS)}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self):
        """Test the classifier on all 26 letters.

        Returns
        -------
        tuple[float, list[dict]]
            (accuracy, results_list) where:
              accuracy     — fraction correct in [0.0, 1.0]
              results_list — one dict per letter with keys:
                               'letter', 'predicted', 'correct', 'confidence'
        """
        results = []
        correct = 0

        for i, letter in enumerate(self.LETTERS):
            one_hot = [0.0] * 26
            one_hot[i] = 1.0

            predicted = self.predict(one_hot)
            prob_dict = self.predict_probs(one_hot)
            confidence = prob_dict[predicted]
            is_correct = predicted == letter

            if is_correct:
                correct += 1

            results.append({
                'letter': letter,
                'predicted': predicted,
                'correct': is_correct,
                'confidence': confidence,
            })

        accuracy = correct / 26
        return accuracy, results


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 phase1-from-scratch/level-a-abcs/07_letter_classifier.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 7 — Letter Classifier Demo")
    print("The first complete working AI: 26-letter recognition")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Build and train
    # -----------------------------------------------------------------------
    clf = LetterClassifier(hidden_size=64, learning_rate=0.05, seed=42)

    print()
    epoch_losses = clf.train(epochs=500, verbose=True)

    # -----------------------------------------------------------------------
    # Evaluate all 26 letters
    # -----------------------------------------------------------------------
    print("\n--- Evaluating all 26 letters ---")
    accuracy, results = clf.evaluate()

    # Print a compact table
    print(f"\n  {'Letter':>6}  {'Predicted':>9}  {'Confidence':>10}  {'Status':>6}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*6}")
    for r in results:
        status = "OK" if r['correct'] else "MISS"
        print(f"  {r['letter']:>6}  {r['predicted']:>9}  "
              f"{r['confidence']:>9.1%}  {status:>6}")

    print(f"\n  Accuracy: {accuracy:.0%}  ({int(accuracy * 26)}/26 correct)")

    # -----------------------------------------------------------------------
    # Probability distributions for a few letters
    # -----------------------------------------------------------------------
    print("\n--- Probability distributions (top-3 per letter) ---")
    for demo_letter in ['a', 'm', 'z']:
        i = clf.LETTERS.index(demo_letter)
        one_hot = [0.0] * 26
        one_hot[i] = 1.0

        probs = clf.predict_probs(one_hot)
        top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_str = "  ".join(f"{l}:{p:.1%}" for l, p in top3)
        print(f"  Input '{demo_letter}' → {top3_str}")

    # -----------------------------------------------------------------------
    # Model size
    # -----------------------------------------------------------------------
    total_params = 0
    for layer in clf.network.layers:
        total_params += len(layer.weights) * len(layer.weights[0])
        total_params += len(layer.biases)

    size_kb = total_params * 4 / 1024  # 4 bytes per float32 equivalent
    print(f"\n--- Model size ---")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Approximate size: {size_kb:.1f} KB  (at 4 bytes/param)")

    print("\n" + "=" * 60)
    print("Chapter 7 complete. The first working AI is alive.")
    print("=" * 60)
