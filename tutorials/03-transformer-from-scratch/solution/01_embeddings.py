"""
Chapter 1 — Embeddings
=======================

What this module teaches
-------------------------
Before a Transformer can process text, it must convert words (discrete symbols)
into continuous vectors (real-valued lists of numbers). This is the job of an
*embedding layer*.

Two key ideas:
  1. **Vocabulary** — a mapping from words to integer indices. Every word gets
     a unique ID. Special tokens handle padding (<pad>), sequence boundaries
     (<sos>, <eos>), and unknown words (<unk>).
  2. **Embedding matrix** — a 2-D table of shape [vocab_size x embed_dim].
     Row i is the vector for token index i. "Looking up" a word's embedding
     means fetching the row at its index.

Why embeddings matter
----------------------
Neural networks operate on numbers, not text. Embeddings map discrete tokens
into a geometric space where *similar meanings sit close together* — something
the network learns during training. Here we initialise them randomly with
Xavier scaling; the Transformer training loop will later refine them.

Builds on: 01_math_foundations.py from level-a-abcs (vector / matrix utilities)
"""

import importlib
import sys
import os
import random
import math

# ---------------------------------------------------------------------------
# Import helpers from Level A
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
math_fn = importlib.import_module('01_math_foundations')


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Maps words to integer indices and back.

    Special tokens are reserved at fixed, low indices so that every model
    trained with this vocabulary shares the same conventions:

        <pad>  → 0   padding — fills short sequences to a fixed length
        <sos>  → 1   start-of-sequence sentinel
        <eos>  → 2   end-of-sequence sentinel
        <unk>  → 3   unknown word not seen during build()

    Usage:
        vocab = Vocabulary()
        vocab.build(["the cat sat", "the dog ran"])
        idx = vocab.encode("cat")          # returns an int
        word = vocab.decode(idx)           # returns "cat"
        ids = vocab.encode_sentence("cat sat")  # [1, idx_cat, idx_sat, 2]
    """

    PAD  = "<pad>"
    SOS  = "<sos>"
    EOS  = "<eos>"
    UNK  = "<unk>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self):
        # word → index
        self._word2idx = {
            self.PAD: self.PAD_IDX,
            self.SOS: self.SOS_IDX,
            self.EOS: self.EOS_IDX,
            self.UNK: self.UNK_IDX,
        }
        # index → word (parallel list for O(1) decode)
        self._idx2word = [self.PAD, self.SOS, self.EOS, self.UNK]

    # ------------------------------------------------------------------
    # Building the vocabulary
    # ------------------------------------------------------------------

    def build(self, texts):
        """Scan a list of strings, tokenise on whitespace, and register every
        unique lowercase word that is not already in the vocabulary.

        Args:
            texts: list[str] — e.g. ["The cat sat", "a dog ran"]

        Returns:
            self  (to allow chaining: vocab = Vocabulary().build(texts))
        """
        for text in texts:
            for raw_word in text.split():
                word = raw_word.lower()
                if word not in self._word2idx:
                    idx = len(self._idx2word)
                    self._word2idx[word] = idx
                    self._idx2word.append(word)
        return self

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, word):
        """Return the integer index for *word* (lowercased).
        Returns UNK_IDX for words not in the vocabulary.
        """
        return self._word2idx.get(word.lower(), self.UNK_IDX)

    def decode(self, index):
        """Return the word string for *index*.
        Returns '<unk>' for out-of-range indices.
        """
        if 0 <= index < len(self._idx2word):
            return self._idx2word[index]
        return self.UNK

    def encode_sentence(self, sentence):
        """Tokenise *sentence*, prepend <sos>, append <eos>, return list of indices.

        Args:
            sentence: str — space-separated words

        Returns:
            list[int] starting with SOS_IDX and ending with EOS_IDX
        """
        tokens = [self.SOS_IDX]
        for raw_word in sentence.split():
            tokens.append(self.encode(raw_word.lower()))
        tokens.append(self.EOS_IDX)
        return tokens

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self):
        """Total number of tokens in the vocabulary (including specials)."""
        return len(self._idx2word)


# ---------------------------------------------------------------------------
# Embedding Layer
# ---------------------------------------------------------------------------

class EmbeddingLayer:
    """A lookup table mapping token indices to dense vectors.

    The embedding matrix has shape [vocab_size x embed_dim].
    Each row is the embedding vector for one token.

    Weights are initialised with Xavier uniform scaling:
        limit = sqrt(6 / (vocab_size + embed_dim))
        w ~ Uniform(-limit, limit)

    This keeps variance reasonable regardless of table dimensions and is a
    standard default before fine-tuning begins.

    Args:
        vocab_size: int — number of tokens (rows in the matrix)
        embed_dim:  int — dimensionality of each embedding vector (cols)
        seed:       int — random seed for reproducibility (default 42)
    """

    def __init__(self, vocab_size, embed_dim, seed=42):
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

        rng = random.Random(seed)

        # Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
        limit = math.sqrt(6.0 / (vocab_size + embed_dim))

        # Build [vocab_size x embed_dim] matrix as a list of lists
        self.weights = [
            [rng.uniform(-limit, limit) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, token_indices):
        """Look up embedding vectors for a sequence of token indices.

        Args:
            token_indices: list[int] — e.g. the output of vocab.encode_sentence()

        Returns:
            list[list[float]] — one vector per token, shape [len x embed_dim]
        """
        return [self.get_embedding(idx) for idx in token_indices]

    def get_embedding(self, index):
        """Return the embedding vector for a single token index.

        Args:
            index: int — token index (0 ≤ index < vocab_size)

        Returns:
            list[float] — length embed_dim
        """
        if not (0 <= index < self.vocab_size):
            raise IndexError(
                f"Token index {index} out of range [0, {self.vocab_size})"
            )
        return list(self.weights[index])  # return a copy to prevent mutation


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a quick brown fox jumps over the lazy dog",
        "transformers use attention to read sentences",
    ]

    print("=" * 60)
    print("Chapter 1: Embeddings Demo")
    print("=" * 60)

    # Build vocabulary
    vocab = Vocabulary().build(sample_sentences)
    print(f"\nVocabulary size: {vocab.size} tokens")
    print(f"  <pad>={vocab.PAD_IDX}, <sos>={vocab.SOS_IDX}, "
          f"<eos>={vocab.EOS_IDX}, <unk>={vocab.UNK_IDX}")

    # Encode / decode
    word = "cat"
    idx  = vocab.encode(word)
    back = vocab.decode(idx)
    print(f"\nEncode '{word}' → {idx} → decode → '{back}'")

    unknown = "elephant"
    print(f"Unknown word '{unknown}' → index {vocab.encode(unknown)} "
          f"(= UNK = {vocab.UNK_IDX})")

    # Encode a sentence
    sentence = "the cat sat"
    ids = vocab.encode_sentence(sentence)
    words_back = [vocab.decode(i) for i in ids]
    print(f"\nSentence: '{sentence}'")
    print(f"  Indices : {ids}")
    print(f"  Decoded : {words_back}")

    # Build embedding layer
    EMBED_DIM = 8
    embed = EmbeddingLayer(vocab.size, EMBED_DIM, seed=42)
    print(f"\nEmbedding matrix: [{vocab.size} x {EMBED_DIM}]")

    # Show embeddings for the encoded sentence
    vectors = embed.forward(ids)
    print(f"\nEmbedding vectors for '{sentence}':")
    for token_id, vec in zip(ids, vectors):
        formatted = [f"{v:+.4f}" for v in vec]
        print(f"  idx={token_id:2d} ({vocab.decode(token_id):>6s}): "
              f"[{', '.join(formatted)}]")
