"""
Level D — Unified Data Pipeline
Loads unified_phonics.json and provides encoding utilities for all architectures.
"""

import json
import random
from pathlib import Path


# Resolve the data file relative to this module, regardless of cwd
_MODULE_DIR = Path(__file__).parent
_DATA_FILE = _MODULE_DIR.parent.parent.parent / "data" / "unified" / "unified_phonics.json"


class UnifiedPhonicsDataset:
    """
    Loads the unified phonics dataset and provides vocabulary, encoding,
    and train/test split utilities used by all Level D architecture chapters.
    """

    def __init__(self, data_path: Path = _DATA_FILE):
        with open(data_path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        self._examples = self._raw["examples"]
        self._input_vocab: dict[str, int] = {}
        self._output_vocab: dict[str, int] = {}
        self._build_vocab()

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def _build_vocab(self) -> None:
        """Collect every unique input character and output phoneme."""
        input_chars: set[str] = set()
        output_phonemes: set[str] = set()

        for ex in self._examples:
            for ch in ex["input"]:
                input_chars.add(ch)

            target = ex["target"]
            if isinstance(target, list):
                for ph in target:
                    if ph:  # skip empty strings (silent e placeholder)
                        output_phonemes.add(ph)
            else:
                if target:
                    output_phonemes.add(target)

        # Deterministic ordering: sort, then assign indices (0 reserved for PAD)
        self._input_vocab = {ch: i + 1 for i, ch in enumerate(sorted(input_chars))}
        self._input_vocab["<PAD>"] = 0

        self._output_vocab = {ph: i + 1 for i, ph in enumerate(sorted(output_phonemes))}
        self._output_vocab["<PAD>"] = 0

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_all_examples(self) -> list[tuple[str, list[str]]]:
        """
        Return every example as (input_str, target_phonemes_list) pairs.
        For letter-type examples the target is wrapped in a list for uniformity.
        """
        results = []
        for ex in self._examples:
            inp = ex["input"]
            target = ex["target"]
            if isinstance(target, str):
                phonemes = [target]
            else:
                phonemes = [ph for ph in target if ph]  # drop empty silent-e markers
            results.append((inp, phonemes))
        return results

    def get_vocab(self) -> tuple[dict[str, int], dict[str, int]]:
        """Return (input_vocab, output_vocab) mapping token -> index."""
        return self._input_vocab, self._output_vocab

    def encode_input(self, text: str) -> list[list[int]]:
        """
        One-hot encode each character of *text*.
        Returns a list of one-hot vectors (list[int]) — one per character.
        Unknown characters map to all-zeros.
        """
        vocab_size = len(self._input_vocab)
        result = []
        for ch in text:
            vec = [0] * vocab_size
            idx = self._input_vocab.get(ch)
            if idx is not None:
                vec[idx] = 1
            result.append(vec)
        return result

    def encode_target(self, phonemes: list[str]) -> list[int]:
        """Convert a list of phoneme strings to a list of vocabulary indices."""
        return [self._output_vocab.get(ph, 0) for ph in phonemes]

    def decode_target(self, indices: list[int]) -> list[str]:
        """Reverse of encode_target — convert indices back to phoneme strings."""
        idx_to_phoneme = {v: k for k, v in self._output_vocab.items()}
        return [idx_to_phoneme.get(i, "<UNK>") for i in indices]

    def train_test_split(
        self, test_ratio: float = 0.2, seed: int = 42
    ) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
        """
        Shuffle deterministically then split into (train, test) example lists.
        test_ratio=0.2 means 80 % train / 20 % test.
        """
        examples = self.get_all_examples()
        rng = random.Random(seed)
        shuffled = examples[:]
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - test_ratio))
        return shuffled[:split_idx], shuffled[split_idx:]

    def summary(self) -> None:
        """Print a concise summary of dataset statistics."""
        examples = self._examples
        type_counts: dict[str, int] = {}
        for ex in examples:
            t = ex["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        print("=" * 50)
        print("Unified Phonics Dataset Summary")
        print("=" * 50)
        print(f"Total examples   : {len(examples)}")
        print(f"Input vocab size : {len(self._input_vocab)}")
        print(f"Output vocab size: {len(self._output_vocab)}")
        print()
        print("Examples by type:")
        for t, count in sorted(type_counts.items()):
            print(f"  {t:<15} {count}")
        print("=" * 50)
