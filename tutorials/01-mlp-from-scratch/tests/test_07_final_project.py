"""Tests for Chapter 7: Letter Classifier (07_letter_classifier.py)

Note: These tests train neural networks, so they run in seconds rather than
milliseconds. That is expected — training is the point of this chapter.
"""

import importlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
classifier_mod = importlib.import_module('07_final_project')

LetterClassifier = classifier_mod.LetterClassifier


# ---------------------------------------------------------------------------
# test_classifier_creation
# ---------------------------------------------------------------------------

def test_classifier_creation():
    """LetterClassifier can be instantiated with default arguments."""
    clf = LetterClassifier()
    assert clf is not None


# ---------------------------------------------------------------------------
# test_classifier_predict_returns_letter
# ---------------------------------------------------------------------------

def test_classifier_predict_returns_letter():
    """predict() returns a single lowercase alphabetic character."""
    clf = LetterClassifier()
    clf.train(epochs=300, verbose=False)
    result = clf.predict([1] + [0] * 25)
    assert isinstance(result, str) and len(result) == 1 and result.isalpha()


# ---------------------------------------------------------------------------
# test_classifier_accuracy_above_95
# ---------------------------------------------------------------------------

def test_classifier_accuracy_above_95():
    """After 500 epochs the classifier achieves >= 95% accuracy (25/26)."""
    clf = LetterClassifier()
    clf.train(epochs=500, verbose=False)
    correct = 0
    for i in range(26):
        one_hot = [0] * 26
        one_hot[i] = 1
        if clf.predict(one_hot) == chr(ord('a') + i):
            correct += 1
    assert correct / 26 >= 0.95, f"Accuracy {correct/26:.0%} below 95%"


# ---------------------------------------------------------------------------
# test_classifier_predict_all_letters
# ---------------------------------------------------------------------------

def test_classifier_predict_all_letters():
    """After training, the classifier produces at least 24 distinct predictions."""
    clf = LetterClassifier()
    clf.train(epochs=500, verbose=False)
    predictions = set()
    for i in range(26):
        one_hot = [0] * 26
        one_hot[i] = 1
        predictions.add(clf.predict(one_hot))
    assert len(predictions) >= 24, (
        f"Only {len(predictions)} unique letters predicted (expected >= 24)"
    )
