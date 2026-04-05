# Unified Phonics Dataset

Combined phonics dataset merging all individual level data sources into a single file for Level D architecture comparison experiments.

## File

`unified_phonics.json` — 129 examples across 5 phonics types.

## Sources

| Source file | Type | Count |
|---|---|---|
| `alphabet/phonemes.json` | letter | 26 |
| `phonics/cvc_words.json` | cvc | 59 |
| `phonics/digraphs.json` | digraph | 22 |
| `phonics/phonics_rules.json` (silent_e) | silent_e | 10 |
| `phonics/phonics_rules.json` (vowel_teams) | vowel_team | 12 |

**Total: 129 examples**

## Schema

Each example has four fields:

- `type` — one of `letter`, `cvc`, `digraph`, `silent_e`, `vowel_team`
- `input` — the written form (single letter or word string)
- `target` — phoneme output: a single phoneme string for `letter` type, or a list of phoneme strings for word types
- `source` — originating data file category

## Usage

Load with `phase1-from-scratch/level-d-unified/01_unified_data.py`:

```python
from phase1_from_scratch.level_d_unified.unified_data import UnifiedPhonicsDataset

ds = UnifiedPhonicsDataset()
ds.summary()
train, test = ds.train_test_split(test_ratio=0.2)
```
