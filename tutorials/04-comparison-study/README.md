# Tutorial 04: Comparison Study

**Put four architectures side by side. See which wins, when, and why.**

---

## What This Tutorial Is

Tutorials 01-03 were step-by-step guides. This one is different — it's a study. You get working implementations of four architectures and a benchmark harness, and the goal is to understand how they compare.

The four architectures:

| File | Architecture | Key Idea |
|------|-------------|----------|
| `01_moe_model.py` | **Mixture of Experts (MoE)** | Route each input to the K most relevant experts; only activate a subset of parameters |
| `02_mamba_model.py` | **Mamba (SSM)** | State-space model: alternative to attention that processes sequences with linear time complexity |
| `03_bitnet_model.py` | **BitNet** | Ternary weights {-1, 0, +1}: 20x smaller than float32, replaces multiplication with addition |
| `04_quantized_transformer.py` | **Quantized Transformer** | Standard Transformer with INT8 weight quantization |

---

## What Makes This Tricky

These four architectures represent four different answers to the same problem: "how do we make neural networks that scale?"

- **MoE** scales by adding experts without activating all of them. GPT-4 is rumored to be an MoE.
- **Mamba** scales by replacing attention (O(n²) with sequence length) with state-space models (O(n)).
- **BitNet** scales by compressing weights to near-zero precision. Microsoft Research published this in 2023.
- **Quantized Transformer** is the pragmatic approach: keep the Transformer, just compress the weights after training.

Each has real tradeoffs. The benchmark suite in `05_benchmark_suite.py` measures accuracy, parameter count, inference speed, and memory footprint.

---

## What Surprised Me

BitNet. The idea that you can replace 32-bit floating point weights with {-1, 0, +1} and still get competitive accuracy seems implausible. But it works — the model keeps full-precision "shadow weights" during training, quantizes them after each update, and uses the Straight-Through Estimator to let gradients flow through the quantization step. The network learns to use ternary weights effectively.

The 20x compression ratio is real: 3,418 float32 parameters would be ~13.4 KB. The same network in ternary is ~1.58 bits per weight — about 675 bytes.

---

## Recommended Approach

1. Read each model file's module docstring to understand the architecture
2. Run each model's demo script to see it train
3. Run the benchmark suite and study the comparison table
4. Try modifying hyperparameters (hidden size, number of experts, etc.) and re-benchmark

---

## Running the Benchmark

```bash
python3 tutorials/04-comparison-study/solution/05_benchmark_suite.py
```

This trains all four architectures on the unified phonics dataset and prints a comparison table. It takes a few minutes.

---

## Architecture Decision Records

The `bonus/architecture-decision-records/` directory has ADR documents explaining the reasoning behind key design choices made across all four tutorials. Worth reading if you want to understand *why* certain choices were made, not just *what* was built.

---

## What's in the Bonus Directory

After this tutorial, there are three bonus topics:

- **`bonus/bitnet-to-c/`** — How to export a trained BitNet model to C code for deployment on embedded hardware (ESP32-class devices)
- **`bonus/arm-qemu-testing/`** — How to verify ARM-compatible code using QEMU via Docker, without owning ARM hardware
- **`bonus/architecture-decision-records/`** — The reasoning behind architectural choices made across all four tutorials
