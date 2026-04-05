"""
Tests for Chapter 6 — Phonics Blender
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup — allow importing from level-b-phonics
# ---------------------------------------------------------------------------
LEVEL_B_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'phase1-from-scratch', 'level-b-phonics'
)
sys.path.insert(0, os.path.normpath(LEVEL_B_DIR))

import importlib

blender_mod = importlib.import_module('06_final_project')
PhonicsBlender = blender_mod.PhonicsBlender


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_blender_creation():
    blender = PhonicsBlender()
    assert blender is not None


def test_blender_blend_returns_phonemes():
    blender = PhonicsBlender()
    blender.train(epochs=200, verbose=False)
    result = blender.blend("cat")
    assert isinstance(result, list)
    assert len(result) == 3


def test_blender_cvc_accuracy():
    blender = PhonicsBlender()
    blender.train(epochs=300, verbose=False)
    test_words = {
        "cat": ["k", "æ", "t"],
        "dog": ["d", "ɒ", "ɡ"],
        "sun": ["s", "ʌ", "n"],
    }
    correct = 0
    total = 0
    for word, expected in test_words.items():
        result = blender.blend(word)
        for r, e in zip(result, expected):
            total += 1
            if r == e:
                correct += 1
    assert correct / total >= 0.7, f"Accuracy {correct/total:.0%} below 70%"


def test_blender_all_words_produce_output():
    blender = PhonicsBlender()
    blender.train(epochs=200, verbose=False)
    for word in ["cat", "dog", "pig", "run", "bed"]:
        result = blender.blend(word)
        assert result is not None and len(result) > 0
