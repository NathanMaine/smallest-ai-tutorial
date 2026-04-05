"""
Tests for 01_embeddings.py

Chapter 1 — Embeddings: Vocabulary and EmbeddingLayer.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_c/test_01_embeddings.py -v
"""

import importlib
import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)
embeddings_mod = importlib.import_module('01_embeddings')

Vocabulary     = embeddings_mod.Vocabulary
EmbeddingLayer = embeddings_mod.EmbeddingLayer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "a quick brown fox jumps over the lazy dog",
]


def _build_vocab():
    return Vocabulary().build(SAMPLE_TEXTS)


# ---------------------------------------------------------------------------
# Vocabulary tests
# ---------------------------------------------------------------------------

def test_vocab_build():
    """Vocabulary built from sentences has size > 4 (more than just specials)."""
    vocab = _build_vocab()
    assert vocab.size > 4, (
        f"Expected size > 4 (at minimum the 4 specials plus real words), "
        f"got {vocab.size}"
    )


def test_vocab_encode_decode():
    """Encoding a known word and decoding the result returns the same word."""
    vocab = _build_vocab()
    word  = "cat"
    idx   = vocab.encode(word)
    back  = vocab.decode(idx)
    assert back == word, f"Round-trip failed: encode('{word}')={idx}, decode({idx})='{back}'"


def test_vocab_special_tokens():
    """Special tokens occupy fixed indices: pad=0, sos=1, eos=2, unk=3."""
    vocab = _build_vocab()
    assert vocab.PAD_IDX == 0
    assert vocab.SOS_IDX == 1
    assert vocab.EOS_IDX == 2
    assert vocab.UNK_IDX == 3
    # Also verify encode() uses the constants
    assert vocab.encode("<pad>") == 0
    assert vocab.encode("<sos>") == 1
    assert vocab.encode("<eos>") == 2
    assert vocab.encode("<unk>") == 3


def test_vocab_unknown_word():
    """A word not seen during build() returns the UNK index."""
    vocab = _build_vocab()
    unk_idx = vocab.encode("xyzzy_not_a_word")
    assert unk_idx == vocab.UNK_IDX, (
        f"Unknown word should return UNK_IDX={vocab.UNK_IDX}, got {unk_idx}"
    )


def test_vocab_encode_sentence():
    """encode_sentence() wraps the token indices with <sos> at start and <eos> at end."""
    vocab = _build_vocab()
    ids   = vocab.encode_sentence("cat sat")
    assert ids[0]  == vocab.SOS_IDX, f"First token should be SOS={vocab.SOS_IDX}, got {ids[0]}"
    assert ids[-1] == vocab.EOS_IDX, f"Last token should be EOS={vocab.EOS_IDX}, got {ids[-1]}"
    # Middle tokens must not be SOS or EOS
    assert len(ids) == 4, f"'cat sat' → [SOS, cat, sat, EOS] = 4 tokens, got {len(ids)}"


def test_vocab_size_property():
    """vocab.size equals the number of unique tokens including specials."""
    vocab = _build_vocab()
    # Manually count unique lowercased words across SAMPLE_TEXTS
    unique_words = set()
    for text in SAMPLE_TEXTS:
        for w in text.split():
            unique_words.add(w.lower())
    expected = len(unique_words) + 4  # +4 for the special tokens
    assert vocab.size == expected, (
        f"Expected vocab.size={expected}, got {vocab.size}"
    )


# ---------------------------------------------------------------------------
# EmbeddingLayer tests
# ---------------------------------------------------------------------------

def test_embedding_shape():
    """forward() returns one vector per token and each vector has length embed_dim."""
    vocab      = _build_vocab()
    embed_dim  = 16
    embed      = EmbeddingLayer(vocab.size, embed_dim, seed=42)
    sentence   = "the cat sat"
    ids        = vocab.encode_sentence(sentence)      # length = 5 (sos + 3 words + eos)
    vectors    = embed.forward(ids)

    assert len(vectors) == len(ids), (
        f"Expected {len(ids)} vectors, got {len(vectors)}"
    )
    for i, vec in enumerate(vectors):
        assert len(vec) == embed_dim, (
            f"Vector {i} has length {len(vec)}, expected {embed_dim}"
        )


def test_embedding_different_tokens():
    """Different token indices produce different embedding vectors."""
    vocab     = _build_vocab()
    embed_dim = 8
    embed     = EmbeddingLayer(vocab.size, embed_dim, seed=42)

    idx_cat = vocab.encode("cat")
    idx_dog = vocab.encode("dog")

    vec_cat = embed.get_embedding(idx_cat)
    vec_dog = embed.get_embedding(idx_dog)

    assert vec_cat != vec_dog, (
        "Embeddings for 'cat' and 'dog' should be different vectors"
    )


def test_embedding_deterministic():
    """Same token index always returns the same embedding vector."""
    vocab     = _build_vocab()
    embed_dim = 8
    embed     = EmbeddingLayer(vocab.size, embed_dim, seed=42)

    idx = vocab.encode("cat")
    v1  = embed.get_embedding(idx)
    v2  = embed.get_embedding(idx)

    assert v1 == v2, "Same index should return the same embedding every time"


def test_embedding_same_seed_reproducible():
    """Two EmbeddingLayer objects with the same seed produce identical matrices."""
    vocab     = _build_vocab()
    embed_dim = 8
    e1 = EmbeddingLayer(vocab.size, embed_dim, seed=0)
    e2 = EmbeddingLayer(vocab.size, embed_dim, seed=0)

    idx = vocab.encode("the")
    assert e1.get_embedding(idx) == e2.get_embedding(idx), (
        "Same seed should produce identical embeddings"
    )


def test_embedding_xavier_range():
    """Xavier-initialised weights should lie within the expected range."""
    import math
    vocab_size = 100
    embed_dim  = 32
    embed      = EmbeddingLayer(vocab_size, embed_dim, seed=7)
    limit      = math.sqrt(6.0 / (vocab_size + embed_dim))

    for row in embed.weights:
        for val in row:
            assert -limit <= val <= limit, (
                f"Weight {val:.6f} outside Xavier range [{-limit:.6f}, {limit:.6f}]"
            )
