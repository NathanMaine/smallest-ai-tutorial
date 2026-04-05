"""
Chapter 6 — Phonics Blender: CVC Word Blending with LSTM
=========================================================

Chapter 6, first sequence-to-sequence model, blending C-A-T into /kæt/.

What this module teaches
-------------------------
We wire a trained LSTM to a real phonics task: given a CVC word letter by
letter (e.g. c-a-t), predict the phoneme produced by each letter
(/k/ /æ/ /t/). This is a sequence-to-sequence problem where every input
position has a corresponding output label.

The model re-uses the LSTMSequenceModel from Chapter 5 unchanged; the new
contribution is the data layer — loading a curated CVC word list, converting
letters to one-hot vectors, and mapping phonemes to class indices.

Key concepts
------------
  One-hot encoding   — a 26-element binary vector representing a letter.
                       Only one position is 1 (the letter's alphabet index);
                       all others are 0. Avoids implying ordinal relationships.

  Phoneme vocabulary — all unique phoneme strings found in the dataset are
                       collected and sorted. Each gets an integer index, so
                       the LSTM output layer has one unit per phoneme.

  Sequence-to-sequence — input length equals output length (3 letters → 3
                          phoneme predictions). No encoder-decoder needed for
                          this short, aligned task.

Chapter roadmap
---------------
  Chapter 1: Recurrence — hidden state, SimpleMemoryCell
  Chapter 2: Vanilla RNN — forward through time and BPTT
  Chapter 3: Vanishing gradients — demonstration and analysis
  Chapter 4: LSTM cell — gated memory (forward only)
  Chapter 5: Trainable LSTM sequence model with full BPTT
  Chapter 6 (this file): Phonics blender — CVC word blending
"""

import importlib
import sys
import os
import json
import random

# ---------------------------------------------------------------------------
# Import LSTM model from Chapter 5 and loss helpers from Chapter 4
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(__file__))

seq_mod = importlib.import_module('05_sequence_model')
LSTMSequenceModel = seq_mod.LSTMSequenceModel

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax


# ---------------------------------------------------------------------------
# PhonicsBlender
# ---------------------------------------------------------------------------

class PhonicsBlender:
    """
    Trains an LSTM to map CVC word letters to their phonemes.

    For each word in the dataset (e.g. "cat"), the model receives three
    one-hot letter vectors in sequence and predicts the phoneme index for
    each position. After training, blend() returns the phoneme string at
    each letter position.

    Parameters
    ----------
    hidden_size    : int  — LSTM hidden units (default 32)
    learning_rate  : float — SGD learning rate (default 0.05)
    seed           : int  — random seed for reproducibility (default 42)
    """

    def __init__(self, hidden_size=32, learning_rate=0.05, seed=42):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.seed = seed

        # ----------------------------------------------------------------
        # Load CVC word data
        # ----------------------------------------------------------------
        data_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'phonics', 'cvc_words.json'
        )
        data_path = os.path.normpath(data_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self.words = raw['words']  # dict: word -> [[letter, phoneme], ...]

        # ----------------------------------------------------------------
        # Build phoneme vocabulary from all phonemes in the data
        # ----------------------------------------------------------------
        all_phonemes = set()
        for pairs in self.words.values():
            for _letter, phoneme in pairs:
                all_phonemes.add(phoneme)

        self.phoneme_vocab = sorted(all_phonemes)
        self.phoneme_to_index = {ph: i for i, ph in enumerate(self.phoneme_vocab)}
        self.index_to_phoneme = {i: ph for ph, i in self.phoneme_to_index.items()}

        # ----------------------------------------------------------------
        # Build LSTM model: 26 inputs (one-hot letter), N outputs (phonemes)
        # ----------------------------------------------------------------
        self.model = LSTMSequenceModel(
            input_size=26,
            hidden_size=hidden_size,
            output_size=len(self.phoneme_vocab),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # One-hot encoding
    # ------------------------------------------------------------------

    def _letter_to_onehot(self, letter):
        """
        Convert a single lower-case letter to a 26-dimensional one-hot vector.

        'a' -> index 0, 'b' -> index 1, ..., 'z' -> index 25.
        Unknown characters map to index 0.
        """
        vec = [0.0] * 26
        idx = ord(letter.lower()) - ord('a')
        if 0 <= idx < 26:
            vec[idx] = 1.0
        return vec

    # ------------------------------------------------------------------
    # Dataset builder
    # ------------------------------------------------------------------

    def _make_dataset(self):
        """
        Convert all CVC words into (input_sequence, targets) training pairs.

        input_sequence : list of 26-dim one-hot vectors (one per letter)
        targets        : list of phoneme indices (one per letter)

        Returns
        -------
        list of (input_sequence, targets) tuples
        """
        dataset = []
        for _word, pairs in self.words.items():
            input_sequence = [self._letter_to_onehot(letter) for letter, _ph in pairs]
            targets = [self.phoneme_to_index[phoneme] for _letter, phoneme in pairs]
            dataset.append((input_sequence, targets))
        return dataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, epochs=200, verbose=True):
        """
        Train the LSTM on all CVC words for the given number of epochs.

        Each epoch shuffles the dataset so the model doesn't overfit to
        word order. Uses the LSTM's built-in SGD update.

        Parameters
        ----------
        epochs  : int  — number of full passes over the dataset
        verbose : bool — print loss every 50 epochs if True

        Returns
        -------
        list of float — average loss per epoch
        """
        dataset = self._make_dataset()
        rng = random.Random(self.seed)

        epoch_losses = []
        for epoch in range(epochs):
            rng.shuffle(dataset)
            total_loss = 0.0
            for sequence, targets in dataset:
                total_loss += self.model.train_step(sequence, targets, lr=self.learning_rate)
            avg_loss = total_loss / len(dataset)
            epoch_losses.append(avg_loss)
            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                print(f"  Epoch {epoch + 1:4d}/{epochs}  loss = {avg_loss:.6f}")

        return epoch_losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def blend(self, word):
        """
        Blend a word into its predicted phonemes.

        Encodes each letter of the word as a one-hot vector, runs the LSTM
        forward pass, and returns the phoneme string predicted at each position.

        Parameters
        ----------
        word : str — lower-case CVC word (e.g. "cat")

        Returns
        -------
        list of str — predicted phoneme at each letter position
                      (e.g. ["k", "æ", "t"])
        """
        sequence = [self._letter_to_onehot(ch) for ch in word.lower()]
        pred_indices = self.model.predict(sequence)
        return [self.index_to_phoneme[idx] for idx in pred_indices]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self):
        """
        Test the blender on all CVC words in the dataset.

        Returns
        -------
        (accuracy, results)
            accuracy : float — fraction of phoneme positions predicted correctly
            results  : list of dicts with keys:
                         word, expected (list), predicted (list), correct (bool)
        """
        results = []
        correct = 0
        total = 0

        for word, pairs in self.words.items():
            expected = [phoneme for _letter, phoneme in pairs]
            predicted = self.blend(word)
            word_correct = (predicted == expected)
            for e, p in zip(expected, predicted):
                total += 1
                if e == p:
                    correct += 1
            results.append({
                'word': word,
                'expected': expected,
                'predicted': predicted,
                'correct': word_correct,
            })

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("Chapter 6 — Phonics Blender: CVC Word Blending with LSTM")
    print("=" * 65)
    print()

    blender = PhonicsBlender(hidden_size=32, learning_rate=0.05, seed=42)

    vocab_size = len(blender.phoneme_vocab)
    param_count = (
        4 * blender.hidden_size * (26 + blender.hidden_size)  # LSTM gate weights
        + 4 * blender.hidden_size                               # LSTM gate biases
        + blender.hidden_size * vocab_size                      # output weights
        + vocab_size                                            # output biases
    )
    print(f"Phoneme vocabulary ({vocab_size} phonemes): {' '.join(blender.phoneme_vocab)}")
    print(f"Model size: ~{param_count:,} parameters")
    print()

    print("Training for 300 epochs...")
    print("-" * 45)
    blender.train(epochs=300, verbose=True)
    print()

    # Show blending of 10 CVC words
    demo_words = ["cat", "dog", "pig", "run", "bed", "hot", "sun", "big", "cup", "hen"]
    expected_map = {word: [ph for _l, ph in pairs] for word, pairs in blender.words.items()}

    print("Blending demo — 10 CVC words:")
    print("-" * 65)
    print(f"  {'Word':<6}  {'Expected':<20}  {'Predicted':<20}  Match")
    print(f"  {'----':<6}  {'--------':<20}  {'---------':<20}  -----")
    for word in demo_words:
        if word not in expected_map:
            continue
        expected = expected_map[word]
        predicted = blender.blend(word)
        match = "YES" if predicted == expected else "no"
        print(f"  {word:<6}  {' '.join(expected):<20}  {' '.join(predicted):<20}  {match}")

    print()
    accuracy, results = blender.evaluate()
    total_words = len(results)
    full_word_correct = sum(1 for r in results if r['correct'])
    print(f"Overall phoneme accuracy : {accuracy:.1%}")
    print(f"Full-word accuracy       : {full_word_correct}/{total_words} words")
    print()
    print("Key insight: each letter position has its own target phoneme,")
    print("so the LSTM learns a sequence-to-sequence mapping —")
    print("the first true seq2seq model in this curriculum.")
