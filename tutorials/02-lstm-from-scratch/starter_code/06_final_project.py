"""
Chapter 6 — Final Project: Phonics Blender
===========================================

Wire the LSTM to a real task: given CVC word letters one at a time
(c-a-t), predict the phoneme at each position (/k/ /æ/ /t/).

This is a sequence-to-sequence problem: every input position has a
corresponding output label. The LSTM reads the full sequence and predicts
at each step.

Your task: implement the PhonicsBlender class to load the data, train
an LSTMSequenceModel on it, and blend() new words.

Goal: achieve >= 80% phoneme accuracy on the CVC training set.

Builds on: 05_sequence_model.py (LSTMSequenceModel)
Data:      data/phonics/cvc_words.json
"""

import importlib
import sys
import os
import json

# Import LSTM model from Chapter 5
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

seq_mod = importlib.import_module('05_sequence_model')
LSTMSequenceModel = seq_mod.LSTMSequenceModel

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax


class PhonicsBlender:
    """
    Trains an LSTM to map CVC word letters to their phonemes.

    For each word in the dataset (e.g. "cat"), the model receives three
    one-hot letter vectors in sequence and predicts the phoneme at each step.

    Parameters
    ----------
    hidden_size   : int   — LSTM hidden units (default 32)
    learning_rate : float — SGD learning rate (default 0.05)
    seed          : int   — random seed (default 42)
    """

    LETTERS = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self, hidden_size=32, learning_rate=0.05, seed=42):
        raise NotImplementedError(
            "Your turn!\n"
            "  1. Load data/phonics/cvc_words.json (path: ../../../data/phonics/cvc_words.json from this file)\n"
            "  2. Build phoneme vocabulary: collect all unique phonemes, sort them, assign integer indices\n"
            "  3. Build an LSTMSequenceModel(input_size=26, output_size=len(phoneme_vocab), ...)\n"
            "  Store: self.words, self.phoneme_vocab, self.phoneme_to_idx, self.model"
        )

    def _letter_to_onehot(self, letter):
        """Convert a letter to a 26-dimensional one-hot vector."""
        vec = [0.0] * 26
        if letter.lower() in self.LETTERS:
            vec[self.LETTERS.index(letter.lower())] = 1.0
        return vec

    def _word_to_sequence(self, word):
        """Convert a word string to a list of one-hot vectors."""
        return [self._letter_to_onehot(ch) for ch in word]

    def train(self, epochs=200, verbose=True):
        """Train on all CVC words in the dataset.

        For each word, create a sequence of one-hot inputs and a list of
        phoneme index targets (one per letter), then call model.train_sequence().

        Returns
        -------
        list[float] — average loss per epoch
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Build training examples from self.words, then train the model.\n"
            "  Each example: (sequence_of_onehots, list_of_phoneme_indices)"
        )

    def blend(self, word):
        """Predict phonemes for each letter in a CVC word.

        Parameters
        ----------
        word : str — 3-letter word (e.g. 'cat')

        Returns
        -------
        list[str] — predicted phoneme at each position (e.g. ['k', 'æ', 't'])
        """
        raise NotImplementedError(
            "Your turn!\n"
            "  Convert word to one-hot sequence, run model.predict_sequence(),\n"
            "  map each predicted index back to the phoneme string."
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    blender = PhonicsBlender(hidden_size=32, learning_rate=0.05, seed=42)
    blender.train(epochs=200)

    for word in ['cat', 'dog', 'pig', 'run', 'bed']:
        phonemes = blender.blend(word)
        print(f"  {word} → {' '.join(phonemes)}")
