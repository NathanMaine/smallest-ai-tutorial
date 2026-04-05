# Setup Guide

This project has exactly two dependencies: Python and pytest.

---

## Python

You need **Python 3.10 or newer**. To check your version:

```bash
python3 --version
```

If you're on macOS or Linux, Python 3 is probably already installed. If not:

- **macOS:** `brew install python`
- **Ubuntu/Debian:** `sudo apt install python3`
- **Windows:** Download from [python.org](https://python.org)

---

## pytest (for running tests)

The tests use pytest. Install it with:

```bash
pip install pytest
```

Or if you want it isolated in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pytest
```

That's it. No other packages needed.

---

## Verifying the Setup

Run the test suite for Tutorial 01 to confirm everything works:

```bash
python3 -m pytest tutorials/01-mlp-from-scratch/tests/ -v
```

You should see output like:

```
tests/test_01_math.py::test_vector_add_basic PASSED
tests/test_01_math.py::test_dot_product_basic PASSED
...
```

If you're running tests against your own `starter_code/` implementations, they'll initially fail with `NotImplementedError` — that's expected. The goal is to make them pass.

---

## Running Solution Files Directly

Every solution file is a standalone script with a built-in demo. You can run any of them directly:

```bash
python3 tutorials/01-mlp-from-scratch/solution/01_math_foundations.py
python3 tutorials/01-mlp-from-scratch/solution/07_final_project.py
```

The final project files train and evaluate a complete model end to end.

---

## Running Tests Against Your Starter Code

As you implement each chapter, run its tests:

```bash
# Tutorial 01
python3 -m pytest tutorials/01-mlp-from-scratch/tests/test_01_math.py -v
python3 -m pytest tutorials/01-mlp-from-scratch/tests/test_02_neuron.py -v
# ...

# All of Tutorial 01 at once
python3 -m pytest tutorials/01-mlp-from-scratch/tests/ -v

# Everything
python3 -m pytest tutorials/ -v
```

---

## File Naming Note

Chapter files are named with numeric prefixes (`01_math_foundations.py`, `02_single_neuron.py`, etc.) because it makes the reading order obvious. Python can't `import` files whose names start with a digit using the normal `import` statement, so the solution files use:

```python
import importlib
module = importlib.import_module('01_math_foundations')
```

This is a minor quirk but it's intentional — it keeps the numbered names while staying valid Python. Each file that does this explains it inline.

---

## Directory Overview

```
smallest-ai-tutorial/
├── README.md               — Start here
├── SETUP.md                — This file
├── CONTRIBUTING.md
├── LICENSE
├── data/
│   └── phonics/            — Training data (CVC words, digraphs, phonics rules)
├── tutorials/
│   ├── 01-mlp-from-scratch/
│   ├── 02-lstm-from-scratch/
│   ├── 03-transformer-from-scratch/
│   └── 04-comparison-study/
└── bonus/
    ├── bitnet-to-c/        — Exporting BitNet weights to C
    ├── arm-qemu-testing/   — Testing on ARM via Docker/QEMU
    └── architecture-decision-records/
```

---

Ready? Start with [Tutorial 01](tutorials/01-mlp-from-scratch/README.md).
