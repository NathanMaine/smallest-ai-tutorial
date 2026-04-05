# Phonics Training Data

This directory contains the training data used across all four tutorials.

---

## Files

### `letters.json`
26 one-hot encoded letter representations and basic letter metadata. This is the primary input format for Tutorial 01 (MLP). Each letter is represented as a 26-dimensional vector with a 1 at its index position.

### `phonemes.json`
Letter-to-phoneme mappings using IPA (International Phonetic Alphabet) notation. Maps each letter of the alphabet to its most common sound(s).

Example:
```json
{
  "a": ["æ", "eɪ", "ɑː"],
  "b": ["b"],
  "c": ["k", "s"]
}
```

The letter 'a' can make three sounds (short 'a' as in "cat", long 'a' as in "cake", broad 'a' as in "father"). This ambiguity is why sequence models (Tutorial 02+) outperform the simple MLP — context resolves it.

### `cvc_words.json`
Consonant-Vowel-Consonant words with letter-by-letter phoneme breakdowns. These are the simplest English words (cat, dog, run) and make good training examples for the LSTM tutorial.

```json
{
  "words": {
    "cat": [["c", "k"], ["a", "æ"], ["t", "t"]],
    "dog": [["d", "d"], ["o", "ɒ"], ["g", "g"]]
  }
}
```

Each word maps to a list of `[letter, phoneme]` pairs.

### `digraphs.json`
Two-letter combinations that make a single sound (sh, ch, th, ph, etc.). These are cases where context matters — 'c' and 'h' individually make their own sounds, but together make /tʃ/. A sequence model can learn this; a single-letter classifier cannot.

### `phonics_rules.json`
A structured set of phonics rules (silent-e, vowel teams, etc.) that explain English pronunciation patterns. These are used as reference material in Tutorial 03 (Transformer) and Tutorial 04 (comparison study).

---

## Why Phonics?

English phonics is a nice domain for this tutorial series because:

1. **Single letters are tractable** — An MLP can classify 26 letters with high accuracy (Tutorial 01)
2. **Context matters** — "ch" and "sh" can't be decoded from one letter (motivates Tutorial 02)
3. **Long-range context helps** — silent-e rules ("cake" vs "cat") require looking ahead (motivates Tutorial 03)
4. **The domain is interpretable** — you can look at predictions and understand why they're right or wrong without domain expertise

The models here are educational demonstrations, not production spelling-to-speech engines. But they're real enough to be interesting.

---

## IPA Notation Quick Reference

If you're not familiar with IPA (International Phonetic Alphabet) symbols:

| IPA | Example word | Sound description |
|-----|-------------|-------------------|
| æ | c**a**t | short 'a' |
| eɪ | c**a**ke | long 'a' |
| ɪ | b**i**g | short 'i' |
| aɪ | b**i**ke | long 'i' |
| ɒ | d**o**g | short 'o' (British) |
| oʊ | b**o**ne | long 'o' |
| ʌ | r**u**n | short 'u' |
| uː | r**u**le | long 'u' |
| tʃ | **ch**ip | 'ch' sound |
| ʃ | **sh**ip | 'sh' sound |
| θ | **th**in | voiceless 'th' |
| ð | **th**is | voiced 'th' |

You don't need to memorize these. The models treat them as class labels — they just need to be consistent.
