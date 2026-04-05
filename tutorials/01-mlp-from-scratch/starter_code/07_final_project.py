"""
Chapter 7 — Final Project: Letter Classifier
=============================================

Everything comes together: the first complete working AI.

A neural network that classifies all 26 letters of the alphabet using only
the foundations built in Chapters 1-6 — no external ML libraries, no NumPy,
no PyTorch. Just pure Python arithmetic and the chain rule.

Your task: implement the LetterClassifier class below.

Architecture
------------
  Input layer   : 26 neurons  (one-hot letter encoding)
  Hidden layer  : 64 neurons  (ReLU activation)
  Output layer  : 26 neurons  (sigmoid activation → softmax for prediction)

  Parameters: 26×64 + 64 + 64×26 + 26 = 3,418
  At 4 bytes each: ~13.4 KB

Training
--------
  Dataset: 26 examples — for each letter i, input = one-hot[i], target = one-hot[i]
  SGD with learning_rate=0.05 converges to >95% accuracy in 500 epochs.

Goal
----
  Implement LetterClassifier so that:
    clf = LetterClassifier()
    clf.train(epochs=500)
    accuracy, _ = clf.evaluate()
    assert accuracy >= 0.95
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
trainer_mod  = importlib.import_module('06_training_loop')
loss_mod     = importlib.import_module('04_loss_function')

BackpropNetwork = backprop_mod.BackpropNetwork
Trainer         = trainer_mod.Trainer
softmax         = loss_mod.softmax


# ---------------------------------------------------------------------------
# LetterClassifier
# ---------------------------------------------------------------------------

class LetterClassifier:
    """A neural network that recognises all 26 letters of the alphabet.

    Attributes
    ----------
    LETTERS       : str             — 'abcdefghijklmnopqrstuvwxyz'
    hidden_size   : int             — number of hidden neurons
    learning_rate : float           — SGD step size
    network       : BackpropNetwork — the underlying neural network
    trainer       : Trainer         — the SGD trainer
    """

    LETTERS = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self, hidden_size=64, learning_rate=0.05, seed=42):
        raise NotImplementedError(
            "Your turn!\n"
            "  Build a BackpropNetwork([26, hidden_size, 26]) and a Trainer.\n"
            "  Store them as self.network and self.trainer."
        )

    def _make_dataset(self):
        """Create 26 training examples — one per letter.

        For each letter i (0-25):
          input  = one-hot vector with a 1 at position i, 0 everywhere else
          target = same one-hot vector

        Returns
        -------
        list[tuple[list[float], list[float]]] — 26 (input, target) pairs
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Create a one-hot vector for each of the 26 letters.\n"
            "  Each vector has length 26, with a 1.0 at index i and 0.0 elsewhere."
        )

    def train(self, epochs=500, verbose=True):
        """Train the classifier on all 26 letters.

        Parameters
        ----------
        epochs  : int  — number of full passes through the 26-letter dataset
        verbose : bool — whether to print loss progress

        Returns
        -------
        list[float] — average loss per epoch
        """
        raise NotImplementedError("Your turn! Use self.trainer.train(dataset, epochs).")

    def predict(self, one_hot_input):
        """Predict the letter for a one-hot input vector.

        Parameters
        ----------
        one_hot_input : list[float] — length-26 one-hot vector

        Returns
        -------
        str — the predicted letter (single lowercase character)
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Run the network forward, apply softmax, find the argmax index,\n"
            "  return self.LETTERS[index]."
        )

    def evaluate(self):
        """Test the classifier on all 26 letters.

        Returns
        -------
        tuple[float, list[dict]]
            (accuracy, results_list) where:
              accuracy  — fraction correct in [0.0, 1.0]
              results   — one dict per letter: 'letter', 'predicted', 'correct', 'confidence'
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Loop over all 26 letters, create one-hot vectors, call predict(),\n"
            "  check if the prediction matches the true letter, return accuracy."
        )


# ---------------------------------------------------------------------------
# Demo (run after implementing LetterClassifier)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 7 — Letter Classifier")
    print("Your first complete AI: 26-letter recognition")
    print("=" * 60)

    clf = LetterClassifier(hidden_size=64, learning_rate=0.05, seed=42)
    clf.train(epochs=500)

    accuracy, results = clf.evaluate()
    print(f"\n  {'Letter':>6}  {'Predicted':>9}  {'Status':>6}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*6}")
    for r in results:
        status = "OK" if r['correct'] else "MISS"
        print(f"  {r['letter']:>6}  {r['predicted']:>9}  {status:>6}")

    print(f"\n  Accuracy: {accuracy:.0%}  ({int(accuracy * 26)}/26 correct)")
    assert accuracy >= 0.95, f"Expected >= 95% accuracy, got {accuracy:.0%}"
    print("\nGoal achieved! >= 95% accuracy.")
