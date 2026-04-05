"""
Tests for Level D — Chapter 5: Quantized Transformer (05_quantized_transformer.py)

Run with: python3 -m pytest tests/test_level_d/test_05_quantized.py -v
"""

import importlib
import sys
import os
import random
import math

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

qt_mod = importlib.import_module('04_quantized_transformer')
quantize_int8 = qt_mod.quantize_int8
dequantize_int8 = qt_mod.dequantize_int8
QuantizedTransformer = qt_mod.QuantizedTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(input_size=8, hidden_size=16, output_size=4,
               num_layers=2, num_heads=2, seed=42):
    return QuantizedTransformer(input_size, hidden_size, output_size,
                                num_layers=num_layers, num_heads=num_heads, seed=seed)


def random_seq(T, input_size, seed=0):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(input_size)] for _ in range(T)]


def random_targets(T, output_size, seed=1):
    rng = random.Random(seed)
    return [rng.randint(0, output_size - 1) for _ in range(T)]


# ---------------------------------------------------------------------------
# quantize_int8 tests
# ---------------------------------------------------------------------------

class TestQuantizeInt8:

    def test_output_values_in_0_255(self):
        """All quantized values must be in [0, 255]."""
        rng = random.Random(7)
        W = [[rng.gauss(0, 2) for _ in range(8)] for _ in range(8)]
        q, s, o = quantize_int8(W)
        for row in q:
            for val in row:
                assert 0 <= val <= 255, f"Out-of-range int8 value: {val}"

    def test_min_weight_maps_to_0(self):
        """The minimum weight should quantize to 0."""
        W = [[0.0, 1.0, -1.0, 2.0]]
        q, s, o = quantize_int8(W)
        # -1.0 is the min — it should be 0
        min_idx = W[0].index(min(W[0]))
        assert q[0][min_idx] == 0, f"Min weight should map to 0, got {q[0][min_idx]}"

    def test_max_weight_maps_to_255(self):
        """The maximum weight should quantize to 255."""
        W = [[0.0, 1.0, -1.0, 2.0]]
        q, s, o = quantize_int8(W)
        max_idx = W[0].index(max(W[0]))
        assert q[0][max_idx] == 255, f"Max weight should map to 255, got {q[0][max_idx]}"

    def test_scale_is_positive(self):
        """Scale must be a positive number."""
        W = [[1.0, 2.0], [-1.0, 3.0]]
        q, s, o = quantize_int8(W)
        assert s > 0, f"Scale must be positive, got {s}"

    def test_offset_equals_min_weight(self):
        """Offset should equal the minimum weight value."""
        W = [[0.5, -0.5, 1.5, -1.0]]
        q, s, o = quantize_int8(W)
        flat = [w for row in W for w in row]
        assert abs(o - min(flat)) < 1e-9, f"Offset {o} != min weight {min(flat)}"

    def test_constant_matrix_safe_scale(self):
        """All-same weights should not crash (zero range → safe scale)."""
        W = [[3.14, 3.14], [3.14, 3.14]]
        q, s, o = quantize_int8(W)
        assert s > 0


# ---------------------------------------------------------------------------
# dequantize_int8 tests
# ---------------------------------------------------------------------------

class TestDequantizeInt8:

    def test_roundtrip_error_small(self):
        """Dequantized values should be close to original (error < 0.05)."""
        rng = random.Random(42)
        W = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(8)]
        q, s, o = quantize_int8(W)
        W_back = dequantize_int8(q, s, o)

        max_err = max(
            abs(W[i][j] - W_back[i][j])
            for i in range(len(W))
            for j in range(len(W[0]))
        )
        assert max_err < 0.05, f"Max round-trip error too large: {max_err:.6f}"

    def test_output_shape_matches_input(self):
        """Dequantized matrix should have the same shape as quantized."""
        W = [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]
        q, s, o = quantize_int8(W)
        W_back = dequantize_int8(q, s, o)
        assert len(W_back) == len(W)
        assert len(W_back[0]) == len(W[0])

    def test_zero_quantized_gives_offset(self):
        """A quantized value of 0 should dequantize to offset."""
        q = [[0, 128]]
        s = 0.1
        o = -5.0
        W_back = dequantize_int8(q, s, o)
        assert abs(W_back[0][0] - o) < 1e-9

    def test_roundtrip_identity_for_exact_range(self):
        """Exact min/max should round-trip without error."""
        W = [[-1.0, 1.0]]
        q, s, o = quantize_int8(W)
        W_back = dequantize_int8(q, s, o)
        # min and max should be reproduced
        assert abs(W_back[0][0] - (-1.0)) < 1e-6
        assert abs(W_back[0][1] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# QuantizedTransformer model tests
# ---------------------------------------------------------------------------

class TestQuantizedTransformerCreation:

    def test_creates_without_error(self):
        model = make_model()
        assert model is not None

    def test_layer_count(self):
        model = make_model(num_layers=2)
        assert len(model.shadow_Q) == 2
        assert len(model.q_Q) == 2

    def test_three_layer_model(self):
        model = make_model(num_layers=3)
        assert len(model.shadow_Q) == 3

    def test_quantized_weights_in_range(self):
        """All quantized weights must be in [0, 255]."""
        model = make_model()
        matrices = (
            [model.q_embed] +
            model.q_Q + model.q_K + model.q_V + model.q_O +
            model.q_W1 + model.q_W2 +
            [model.q_out]
        )
        for mat in matrices:
            for row in mat:
                for val in row:
                    assert 0 <= val <= 255, f"Out-of-range int8 value: {val}"


class TestQuantizedTransformerForward:

    def test_forward_returns_correct_output_size(self):
        model = make_model(input_size=8, hidden_size=16, output_size=4)
        x = [0.5] * 8
        out = model.forward(x)
        assert len(out) == 4

    def test_forward_sequence_returns_T_outputs(self):
        model = make_model(input_size=8, hidden_size=16, output_size=4)
        seq = random_seq(6, 8)
        outs = model.forward_sequence(seq)
        assert len(outs) == 6
        for o in outs:
            assert len(o) == 4

    def test_forward_is_deterministic(self):
        """Same input should give same output."""
        model = make_model(seed=11)
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        out1 = model.forward(x)
        out2 = model.forward(x)
        assert out1 == out2

    def test_forward_output_is_finite(self):
        """All output values should be finite floats."""
        model = make_model()
        x = random_seq(1, 8)[0]
        out = model.forward(x)
        for val in out:
            assert math.isfinite(val), f"Non-finite output: {val}"


class TestQuantizedTransformerTraining:

    def test_training_reduces_loss(self):
        """Loss should decrease after several training steps."""
        model = make_model(input_size=8, hidden_size=16, output_size=4, seed=0)
        seq = random_seq(6, 8, seed=5)
        targets = random_targets(6, 4, seed=6)

        loss_first = model.train_step(seq, targets, lr=0.05)
        for _ in range(20):
            loss = model.train_step(seq, targets, lr=0.05)

        assert loss < loss_first, (
            f"Loss did not decrease: initial={loss_first:.4f}, final={loss:.4f}"
        )

    def test_train_step_returns_scalar(self):
        model = make_model()
        seq = random_seq(4, 8)
        targets = random_targets(4, 4)
        loss = model.train_step(seq, targets, lr=0.01)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_loss_finite(self):
        model = make_model()
        seq = random_seq(3, 8)
        targets = random_targets(3, 4)
        loss = model.train_step(seq, targets, lr=0.01)
        assert math.isfinite(loss), f"Loss is not finite: {loss}"


class TestQuantizedTransformerPredict:

    def test_predict_returns_valid_indices(self):
        """All predictions must be valid class indices."""
        model = make_model(input_size=8, hidden_size=16, output_size=4)
        seq = random_seq(5, 8)
        preds = model.predict(seq)
        assert len(preds) == 5
        for p in preds:
            assert 0 <= p < 4, f"Invalid prediction index: {p}"

    def test_predict_returns_integers(self):
        model = make_model()
        seq = random_seq(3, 8)
        preds = model.predict(seq)
        for p in preds:
            assert isinstance(p, int)


class TestQuantizedTransformerCompressedSize:

    def test_params_count_positive(self):
        model = make_model(input_size=8, hidden_size=16, output_size=4)
        assert model.get_params_count() > 0

    def test_compressed_size_equals_params_bytes(self):
        """Int8 = 1 byte per parameter."""
        model = make_model(input_size=8, hidden_size=16, output_size=4)
        assert model.get_compressed_size_bytes() == model.get_params_count()

    def test_compressed_size_much_smaller_than_float32(self):
        """Int8 storage should be 4x smaller than float32."""
        model = make_model(input_size=8, hidden_size=16, output_size=4)
        float_bytes = model.get_params_count() * 4
        int8_bytes = model.get_compressed_size_bytes()
        assert int8_bytes * 4 == float_bytes, (
            f"Int8 bytes ({int8_bytes}) × 4 != float32 bytes ({float_bytes})"
        )
