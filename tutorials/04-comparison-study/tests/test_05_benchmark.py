"""
Tests for Level D — Chapter 6: Benchmark Suite (06_benchmark_suite.py)

Run with: python3 -m pytest tests/test_level_d/test_06_benchmark.py -v
"""

import importlib
import sys
import os
import json
import tempfile
import random

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..', '01-mlp-from-scratch', 'solution')
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)

bench_mod = importlib.import_module('05_benchmark_suite')
BenchmarkSuite = bench_mod.BenchmarkSuite


# ---------------------------------------------------------------------------
# Minimal stub model — avoids running real (slow) architectures in unit tests
# ---------------------------------------------------------------------------

class _StubModel:
    """Tiny stub that satisfies the Level D model interface."""

    def __init__(self, input_size, output_size, seed=0):
        self.input_size  = input_size
        self.output_size = output_size
        self._rng = random.Random(seed)
        self._params = input_size * output_size + output_size

    def train_step(self, input_seq, targets, lr):
        # Always return a small positive loss
        return 1.0

    def predict(self, input_seq):
        return [self._rng.randint(0, self.output_size - 1) for _ in input_seq]

    def get_params_count(self):
        return self._params

    def get_compressed_size_bytes(self):
        # Simulated 2x compression
        return self._params * 2.0


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_suite():
    return BenchmarkSuite()


def make_stub(suite):
    return _StubModel(suite.input_size, suite.output_size)


# ---------------------------------------------------------------------------
# BenchmarkSuite initialisation
# ---------------------------------------------------------------------------

class TestBenchmarkSuiteInit:

    def test_creates_without_error(self):
        suite = make_suite()
        assert suite is not None

    def test_has_train_data(self):
        suite = make_suite()
        assert len(suite.train_data) > 0

    def test_has_test_data(self):
        suite = make_suite()
        assert len(suite.test_data) > 0

    def test_input_size_positive(self):
        suite = make_suite()
        assert suite.input_size > 0

    def test_output_size_positive(self):
        suite = make_suite()
        assert suite.output_size > 0


# ---------------------------------------------------------------------------
# benchmark_model — return dict shape
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"name", "accuracy", "total_params",
                 "compressed_bytes", "inference_ms", "memory_bytes"}


class TestBenchmarkModelReturnShape:

    def test_returns_dict(self):
        suite = make_suite()
        model = make_stub(suite)
        result = suite.benchmark_model(model, "Stub", train_epochs=1,
                                       num_inference_runs=2)
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        suite = make_suite()
        model = make_stub(suite)
        result = suite.benchmark_model(model, "Stub", train_epochs=1,
                                       num_inference_runs=2)
        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_KEYS - result.keys()}"
        )

    def test_name_matches_argument(self):
        suite = make_suite()
        model = make_stub(suite)
        result = suite.benchmark_model(model, "TestArch", train_epochs=1,
                                       num_inference_runs=2)
        assert result["name"] == "TestArch"


# ---------------------------------------------------------------------------
# benchmark_model — metric values are valid
# ---------------------------------------------------------------------------

class TestBenchmarkModelMetricValues:

    def _get_result(self):
        suite = make_suite()
        model = make_stub(suite)
        return suite.benchmark_model(model, "Stub", train_epochs=1,
                                     num_inference_runs=2)

    def test_accuracy_between_0_and_1(self):
        result = self._get_result()
        assert 0.0 <= result["accuracy"] <= 1.0, (
            f"Accuracy out of range: {result['accuracy']}"
        )

    def test_total_params_positive(self):
        result = self._get_result()
        assert result["total_params"] > 0

    def test_compressed_bytes_positive(self):
        result = self._get_result()
        assert result["compressed_bytes"] > 0

    def test_inference_ms_positive(self):
        result = self._get_result()
        assert result["inference_ms"] > 0, (
            f"inference_ms must be positive: {result['inference_ms']}"
        )

    def test_memory_bytes_positive(self):
        result = self._get_result()
        assert result["memory_bytes"] > 0

    def test_all_numeric(self):
        result = self._get_result()
        for key in ("accuracy", "total_params", "compressed_bytes",
                    "inference_ms", "memory_bytes"):
            assert isinstance(result[key], (int, float)), (
                f"{key} is not numeric: {type(result[key])}"
            )


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------

class TestFormatTable:

    def _fake_results(self):
        return [
            {"name": "MoE",    "accuracy": 0.5,  "total_params": 1000,
             "compressed_bytes": 4000.0, "inference_ms": 0.5, "memory_bytes": 4000.0},
            {"name": "BitNet", "accuracy": 0.45, "total_params": 800,
             "compressed_bytes": 158.0,  "inference_ms": 0.3, "memory_bytes": 158.0},
        ]

    def test_returns_string(self):
        suite = make_suite()
        table = suite.format_table(self._fake_results())
        assert isinstance(table, str)

    def test_non_empty_string(self):
        suite = make_suite()
        table = suite.format_table(self._fake_results())
        assert len(table.strip()) > 0

    def test_contains_architecture_names(self):
        suite = make_suite()
        table = suite.format_table(self._fake_results())
        assert "MoE" in table
        assert "BitNet" in table

    def test_contains_accuracy_values(self):
        suite = make_suite()
        table = suite.format_table(self._fake_results())
        # Some representation of the accuracy should appear
        assert "0.5" in table or "0.45" in table

    def test_multiline(self):
        suite = make_suite()
        table = suite.format_table(self._fake_results())
        assert "\n" in table


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:

    def _fake_results(self):
        return [
            {"name": "Stub", "accuracy": 0.42, "total_params": 500,
             "compressed_bytes": 1000.0, "inference_ms": 0.2, "memory_bytes": 1000.0}
        ]

    def test_creates_file(self):
        suite = make_suite()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        suite.save_results(self._fake_results(), path)
        assert os.path.exists(path)

    def test_file_is_valid_json(self):
        suite = make_suite()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        suite.save_results(self._fake_results(), path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_saved_data_matches_input(self):
        suite = make_suite()
        fake = self._fake_results()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        suite.save_results(fake, path)
        with open(path) as f:
            data = json.load(f)
        assert data[0]["name"] == "Stub"
        assert abs(data[0]["accuracy"] - 0.42) < 1e-9

    def test_file_not_empty(self):
        suite = make_suite()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        suite.save_results(self._fake_results(), path)
        assert os.path.getsize(path) > 0
