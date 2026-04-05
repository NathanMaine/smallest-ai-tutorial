"""
Tests for Level D — Chapter 4: BitNet Model (04_bitnet_model.py)

Run with: python3 -m pytest tests/test_level_d/test_04_bitnet.py -v
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

bitnet_mod = importlib.import_module('03_bitnet_model')
quantize_ternary = bitnet_mod.quantize_ternary
ternary_matmul = bitnet_mod.ternary_matmul
BitNetModel = bitnet_mod.BitNetModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(input_size=8, hidden_size=16, output_size=4, num_layers=2, seed=42):
    return BitNetModel(input_size, hidden_size, output_size, num_layers=num_layers, seed=seed)


def random_seq(T, input_size, seed=0):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(input_size)] for _ in range(T)]


def random_targets(T, output_size, seed=1):
    rng = random.Random(seed)
    return [rng.randint(0, output_size - 1) for _ in range(T)]


# ---------------------------------------------------------------------------
# quantize_ternary tests
# ---------------------------------------------------------------------------

class TestQuantizeTernary:

    def test_output_values_only_ternary(self):
        """All quantized values must be in {-1, 0, 1}."""
        rng = random.Random(99)
        W = [[rng.gauss(0, 1) for _ in range(10)] for _ in range(10)]
        q, s = quantize_ternary(W)
        for row in q:
            for val in row:
                assert val in (-1, 0, 1), f"Unexpected value: {val}"

    def test_threshold_zeros_small_weights(self):
        """Weights below threshold must map to 0."""
        W = [[0.1, -0.2, 0.3]]
        q, s = quantize_ternary(W, threshold=0.5)
        assert q[0] == [0, 0, 0], f"Expected all zeros, got {q[0]}"

    def test_positive_large_weights_map_to_plus1(self):
        """Positive weights above threshold → +1."""
        W = [[0.8, 1.5, 2.0]]
        q, s = quantize_ternary(W, threshold=0.5)
        assert q[0] == [1, 1, 1], f"Expected [1,1,1], got {q[0]}"

    def test_negative_large_weights_map_to_minus1(self):
        """Negative weights below -threshold → -1."""
        W = [[-0.8, -1.5, -2.0]]
        q, s = quantize_ternary(W, threshold=0.5)
        assert q[0] == [-1, -1, -1], f"Expected [-1,-1,-1], got {q[0]}"

    def test_scale_is_positive(self):
        """Scale factor must be a positive number."""
        W = [[0.9, -0.7, 0.6, -1.2]]
        q, s = quantize_ternary(W)
        assert s > 0, f"Scale must be positive, got {s}"

    def test_scale_is_mean_abs_surviving(self):
        """Scale should equal mean |w| for weights where |w| >= threshold."""
        W = [[0.8, -0.6, 0.1, -0.9]]
        q, s = quantize_ternary(W, threshold=0.5)
        # Surviving: |0.8|=0.8, |-0.6|=0.6, |-0.9|=0.9  (0.1 < 0.5 → zero)
        expected_scale = (0.8 + 0.6 + 0.9) / 3
        assert abs(s - expected_scale) < 1e-9, f"Expected scale {expected_scale:.4f}, got {s:.4f}"

    def test_mixed_sign_and_zero(self):
        """Verify correct sign assignment for mixed-sign weights."""
        W = [[0.7, -0.8, 0.0, 0.3, -1.0]]
        q, s = quantize_ternary(W, threshold=0.5)
        assert q[0][0] == 1    # 0.7 > threshold
        assert q[0][1] == -1   # -0.8 < -threshold
        assert q[0][2] == 0    # 0.0 < threshold
        assert q[0][3] == 0    # 0.3 < threshold
        assert q[0][4] == -1   # -1.0 < -threshold

    def test_all_zeros_returns_safe_scale(self):
        """If no weights survive quantization, scale should be a safe small value."""
        W = [[0.1, 0.2, -0.1]]
        q, s = quantize_ternary(W, threshold=0.5)
        assert s > 0, f"Scale must be positive even when all weights zero out, got {s}"
        # All weights should be zero
        assert all(v == 0 for v in q[0])


# ---------------------------------------------------------------------------
# ternary_matmul tests
# ---------------------------------------------------------------------------

class TestTernaryMatmul:

    def test_output_length_matches_rows(self):
        """Output vector length must equal number of rows in weight matrix."""
        q = [[1, -1, 0], [0, 1, -1], [1, 0, 1], [-1, -1, 0]]  # 4×3
        scale = 0.5
        x = [1.0, 2.0, 3.0]
        out = ternary_matmul(q, scale, x)
        assert len(out) == 4

    def test_matches_regular_matmul_approximately(self):
        """Ternary matmul with scale 1.0 on an integer weight matrix should be exact."""
        # All-+1 weights: output[i] = sum(x)
        q = [[1, 1, 1], [1, 1, 1]]
        scale = 1.0
        x = [1.0, 2.0, 3.0]
        out = ternary_matmul(q, scale, x)
        assert abs(out[0] - 6.0) < 1e-9
        assert abs(out[1] - 6.0) < 1e-9

    def test_subtraction_for_minus1_weights(self):
        """Weights of -1 should subtract their input contribution."""
        q = [[-1, -1, -1]]
        scale = 1.0
        x = [1.0, 2.0, 3.0]
        out = ternary_matmul(q, scale, x)
        assert abs(out[0] - (-6.0)) < 1e-9

    def test_zero_weights_have_no_effect(self):
        """Zero weights should contribute nothing."""
        q = [[0, 0, 0]]
        scale = 2.0
        x = [10.0, 20.0, 30.0]
        out = ternary_matmul(q, scale, x)
        assert abs(out[0]) < 1e-9

    def test_scale_multiplied_into_result(self):
        """The scale factor must multiply the raw accumulation."""
        q = [[1, 1]]
        scale = 3.5
        x = [1.0, 1.0]
        out = ternary_matmul(q, scale, x)
        assert abs(out[0] - 7.0) < 1e-9  # (1+1) * 3.5

    def test_approximates_float_matmul(self):
        """After quantizing a float matrix, ternary_matmul should be close to float matmul."""
        rng = random.Random(7)
        W = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(8)]
        x = [rng.gauss(0, 1) for _ in range(8)]
        q, s = quantize_ternary(W, threshold=0.2)

        # Regular matmul
        expected = []
        for row in W:
            val = sum(wi * xi for wi, xi in zip(row, x))
            expected.append(val)

        got = ternary_matmul(q, s, x)

        # Not exact, but should be correlated — check at least same sign for most
        matching_signs = sum(
            1 for e, g in zip(expected, got)
            if (e >= 0) == (g >= 0)
        )
        assert matching_signs >= 5, (
            f"Too few matching signs ({matching_signs}/8); quantization may be wrong"
        )


# ---------------------------------------------------------------------------
# BitNetModel tests
# ---------------------------------------------------------------------------

class TestBitNetModelCreation:

    def test_creates_without_error(self):
        model = make_model()
        assert model is not None

    def test_default_layer_count(self):
        model = make_model(num_layers=2)
        assert len(model.shadow_weights) == 2
        assert len(model.quantized_weights) == 2

    def test_three_layer_model(self):
        model = make_model(input_size=8, hidden_size=16, output_size=4, num_layers=3)
        assert len(model.shadow_weights) == 3

    def test_shadow_weights_shapes(self):
        model = make_model(input_size=8, hidden_size=16, output_size=4, num_layers=2)
        # Layer 0: hidden×input
        assert len(model.shadow_weights[0]) == 16
        assert len(model.shadow_weights[0][0]) == 8
        # Layer 1: output×hidden
        assert len(model.shadow_weights[1]) == 4
        assert len(model.shadow_weights[1][0]) == 16


class TestWeightsTernaryAfterInit:

    def test_quantized_values_are_ternary(self):
        """After init, all quantized weights must be in {-1, 0, 1}."""
        model = make_model()
        for q_matrix in model.quantized_weights:
            for row in q_matrix:
                for val in row:
                    assert val in (-1, 0, 1), f"Non-ternary value: {val}"

    def test_scales_are_positive(self):
        model = make_model()
        for s in model.scales:
            assert s > 0


class TestBitNetForward:

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
        model = make_model(seed=123)
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        out1 = model.forward(x)
        out2 = model.forward(x)
        assert out1 == out2


class TestBitNetTraining:

    def test_training_reduces_loss(self):
        """Loss should generally decrease over several training steps."""
        model = make_model(input_size=8, hidden_size=32, output_size=4, seed=42)
        seq = random_seq(8, 8, seed=5)
        targets = random_targets(8, 4, seed=6)

        loss_first = model.train_step(seq, targets, lr=0.05)
        for _ in range(30):
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


class TestBitNetPredict:

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


class TestBitNetCompressedSize:

    def test_params_count_positive(self):
        model = make_model(input_size=8, hidden_size=16, output_size=4)
        assert model.get_params_count() > 0

    def test_compressed_size_much_smaller_than_float(self):
        """Compressed size should be much smaller than float32 storage."""
        model = make_model(input_size=8, hidden_size=16, output_size=4)
        float_bytes = model.get_params_count() * 4
        compressed_bytes = model.get_compressed_size_bytes()
        # Ternary weights need ~1.58/32 ≈ 5% of float32 space
        assert compressed_bytes < float_bytes * 0.5, (
            f"Compressed ({compressed_bytes:.1f}B) is not much smaller than "
            f"float32 ({float_bytes:.1f}B)"
        )

    def test_compressed_size_is_positive(self):
        model = make_model()
        assert model.get_compressed_size_bytes() > 0
