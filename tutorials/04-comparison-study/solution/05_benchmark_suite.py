"""
Chapter 6 — Benchmark Suite
============================

Provides a unified benchmarking framework that evaluates all four Level D
architectures (MoE, Mamba, BitNet, QuantizedTransformer) on the same
phonics dataset using consistent metrics:

  - Accuracy      : phoneme-level prediction accuracy on the test split
  - Total params  : number of scalar parameters in the model
  - Compressed bytes : storage size (architecture-specific compression)
  - Inference speed  : average milliseconds per forward pass
  - Memory bytes     : estimated RAM usage during inference

Usage
-----
    suite = BenchmarkSuite()
    results = suite.benchmark_all()
    print(suite.format_table(results))
    suite.save_results(results, "benchmark_results.json")
"""

import importlib
import sys
import os
import time
import json

# ---------------------------------------------------------------------------
# Path setup — Level A and Level D modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

data_mod = importlib.import_module('01_unified_data')
UnifiedPhonicsDataset = data_mod.UnifiedPhonicsDataset

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """
    Loads the unified phonics dataset once and benchmarks any model that
    exposes the standard Level D interface:

        model.train_step(input_seq, targets, lr)  → float loss
        model.predict(input_seq)                  → list[int] predictions
        model.get_params_count()                  → int
        model.get_compressed_size_bytes()         → float  (optional)
    """

    def __init__(self):
        self.dataset = UnifiedPhonicsDataset()
        self.train_data, self.test_data = self.dataset.train_test_split()
        _, self.output_vocab = self.dataset.get_vocab()
        self.input_size = len(self.dataset.get_vocab()[0])   # input vocab size (incl. PAD)
        self.output_size = len(self.output_vocab)            # output vocab size (incl. PAD)

    # ------------------------------------------------------------------
    # Core benchmark
    # ------------------------------------------------------------------

    def benchmark_model(self, model, name, train_epochs=50, lr=0.05,
                        num_inference_runs=5):
        """Train model, evaluate it, and collect performance metrics.

        Parameters
        ----------
        model        : any Level D model instance
        name         : str — human-readable architecture name
        train_epochs : int — number of full passes over the training set
        lr           : float — learning rate for training
        num_inference_runs : int — number of runs for timing average

        Returns
        -------
        dict with keys: name, accuracy, total_params, compressed_bytes,
                        inference_ms, memory_bytes
        """
        # ---- Training ----
        for _ in range(train_epochs):
            for text, phonemes in self.train_data:
                input_seq = self.dataset.encode_input(text)
                targets = self.dataset.encode_target(phonemes)
                if not input_seq or not targets:
                    continue
                # Align lengths (some examples may have mismatched lengths)
                min_len = min(len(input_seq), len(targets))
                model.train_step(input_seq[:min_len], targets[:min_len], lr)

        # ---- Accuracy evaluation on test set ----
        correct = 0
        total = 0
        for text, phonemes in self.test_data:
            input_seq = self.dataset.encode_input(text)
            targets = self.dataset.encode_target(phonemes)
            if not input_seq or not targets:
                continue
            min_len = min(len(input_seq), len(targets))
            input_seq = input_seq[:min_len]
            targets = targets[:min_len]

            preds = model.predict(input_seq)
            for p, t in zip(preds, targets):
                if p == t:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        # ---- Model size ----
        total_params = model.get_params_count()
        if hasattr(model, 'get_compressed_size_bytes'):
            compressed_bytes = model.get_compressed_size_bytes()
        else:
            compressed_bytes = total_params * 4  # float32 fallback

        # ---- Inference speed (average over num_inference_runs) ----
        # Use the first test example for timing; fall back to a dummy if empty
        if self.test_data:
            sample_text, sample_phonemes = self.test_data[0]
            sample_input = self.dataset.encode_input(sample_text)
        else:
            sample_input = [[0.0] * self.input_size]

        times = []
        for _ in range(num_inference_runs):
            t0 = time.perf_counter()
            model.predict(sample_input)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        inference_ms = (sum(times) / len(times)) * 1000.0

        # ---- Memory estimate ----
        # Use compressed size if available, otherwise params * 4 bytes (float32)
        memory_bytes = compressed_bytes

        return {
            "name": name,
            "accuracy": accuracy,
            "total_params": total_params,
            "compressed_bytes": compressed_bytes,
            "inference_ms": inference_ms,
            "memory_bytes": memory_bytes,
        }

    # ------------------------------------------------------------------
    # Benchmark all four architectures
    # ------------------------------------------------------------------

    def benchmark_all(self):
        """Instantiate and benchmark all four Level D architectures.

        Returns
        -------
        list[dict] — one result dict per architecture
        """
        moe_mod    = importlib.import_module('01_moe_model')
        mamba_mod  = importlib.import_module('02_mamba_model')
        bitnet_mod = importlib.import_module('03_bitnet_model')
        quant_mod  = importlib.import_module('04_quantized_transformer')

        hidden = 32  # shared hidden size — small enough for fast tests

        models = [
            (
                moe_mod.MoEModel(
                    input_size=self.input_size,
                    hidden_size=hidden,
                    output_size=self.output_size,
                    num_experts=4,
                    top_k=2,
                    seed=42,
                ),
                "MoE",
            ),
            (
                mamba_mod.MambaModel(
                    input_size=self.input_size,
                    hidden_size=hidden,
                    output_size=self.output_size,
                    state_dim=16,
                    seed=42,
                ),
                "Mamba",
            ),
            (
                bitnet_mod.BitNetModel(
                    input_size=self.input_size,
                    hidden_size=hidden,
                    output_size=self.output_size,
                    num_layers=2,
                    seed=42,
                ),
                "BitNet",
            ),
            (
                quant_mod.QuantizedTransformer(
                    input_size=self.input_size,
                    hidden_size=hidden,
                    output_size=self.output_size,
                    num_layers=2,
                    num_heads=2,
                    seed=42,
                ),
                "QuantizedTransformer",
            ),
        ]

        results = []
        for model, name in models:
            result = self.benchmark_model(model, name)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def format_table(self, results):
        """Format benchmark results as a human-readable ASCII table.

        Parameters
        ----------
        results : list[dict] — output of benchmark_model / benchmark_all

        Returns
        -------
        str — multi-line ASCII table
        """
        header = (
            f"{'Architecture':<24} {'Accuracy':>9} {'Params':>10} "
            f"{'Size(B)':>12} {'Inf(ms)':>10} {'Mem(B)':>12}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for r in results:
            line = (
                f"{r['name']:<24} "
                f"{r['accuracy']:>9.4f} "
                f"{r['total_params']:>10,} "
                f"{r['compressed_bytes']:>12,.1f} "
                f"{r['inference_ms']:>10.4f} "
                f"{r['memory_bytes']:>12,.1f}"
            )
            lines.append(line)

        lines.append(sep)
        return "\n".join(lines)

    def save_results(self, results, path):
        """Save benchmark results as a JSON file.

        Parameters
        ----------
        results : list[dict]
        path    : str — file path to write
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 6 — Benchmark Suite")
    print("=" * 60)

    suite = BenchmarkSuite()
    print(f"\nDataset: {len(suite.train_data)} train / {len(suite.test_data)} test examples")
    print(f"Input vocab size : {suite.input_size}")
    print(f"Output vocab size: {suite.output_size}")

    print("\nRunning benchmarks (this may take a moment)...")
    results = suite.benchmark_all()

    print("\n" + suite.format_table(results))
    suite.save_results(results, "/tmp/benchmark_results.json")
    print("\nResults saved to /tmp/benchmark_results.json")
