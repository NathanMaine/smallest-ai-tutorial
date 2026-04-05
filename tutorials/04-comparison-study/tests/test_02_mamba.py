"""
Tests for Level D — Mamba / Selective SSM Model (03_mamba_model.py)

Run with: python3 -m pytest tests/test_level_d/test_03_mamba.py -v
"""

import importlib
import sys
import os
import random

# ---- Module loading --------------------------------------------------------
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..', '01-mlp-from-scratch', 'solution')
)

mamba_mod = importlib.import_module('02_mamba_model')
MambaModel = mamba_mod.MambaModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_SIZE = 8
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4
STATE_DIM = 16


def make_model(seed=42):
    return MambaModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                      state_dim=STATE_DIM, seed=seed)


def one_hot(idx, size):
    v = [0.0] * size
    v[idx] = 1.0
    return v


def make_sequence(length=5, seed=7):
    rng = random.Random(seed)
    seq = [one_hot(rng.randint(0, INPUT_SIZE - 1), INPUT_SIZE) for _ in range(length)]
    targets = [rng.randint(0, OUTPUT_SIZE - 1) for _ in range(length)]
    return seq, targets


# ---------------------------------------------------------------------------
# Test: creation
# ---------------------------------------------------------------------------

class TestMambaCreation:
    def test_model_instantiates(self):
        model = make_model()
        assert model is not None

    def test_state_initialized_to_zeros(self):
        model = make_model()
        assert all(s == 0.0 for s in model.state)

    def test_state_dim_correct(self):
        model = make_model()
        assert len(model.state) == STATE_DIM

    def test_matrices_have_correct_shapes(self):
        model = make_model()
        # A: state_dim × state_dim
        assert len(model.A) == STATE_DIM
        assert len(model.A[0]) == STATE_DIM
        # B: state_dim × input_size
        assert len(model.B) == STATE_DIM
        assert len(model.B[0]) == INPUT_SIZE
        # C: hidden_size × state_dim
        assert len(model.C) == HIDDEN_SIZE
        assert len(model.C[0]) == STATE_DIM
        # W_out: output_size × hidden_size
        assert len(model.W_out) == OUTPUT_SIZE
        assert len(model.W_out[0]) == HIDDEN_SIZE


# ---------------------------------------------------------------------------
# Test: forward_step
# ---------------------------------------------------------------------------

class TestMambaForwardStep:
    def test_forward_step_returns_correct_output_size(self):
        model = make_model()
        x = one_hot(3, INPUT_SIZE)
        out = model.forward_step(x)
        assert len(out) == OUTPUT_SIZE

    def test_forward_step_returns_list_of_floats(self):
        model = make_model()
        x = one_hot(0, INPUT_SIZE)
        out = model.forward_step(x)
        assert all(isinstance(v, float) for v in out)

    def test_forward_step_changes_state(self):
        model = make_model()
        state_before = list(model.state)
        x = one_hot(2, INPUT_SIZE)
        model.forward_step(x)
        state_after = list(model.state)
        assert state_before != state_after, "State should change after forward_step"

    def test_forward_step_state_not_all_zero_after_step(self):
        model = make_model()
        x = one_hot(1, INPUT_SIZE)
        model.forward_step(x)
        assert any(abs(s) > 1e-10 for s in model.state)


# ---------------------------------------------------------------------------
# Test: forward (sequence)
# ---------------------------------------------------------------------------

class TestMambaForward:
    def test_forward_returns_correct_number_of_outputs(self):
        model = make_model()
        seq, _ = make_sequence(length=5)
        outputs = model.forward(seq)
        assert len(outputs) == 5

    def test_forward_each_output_correct_size(self):
        model = make_model()
        seq, _ = make_sequence(length=4)
        outputs = model.forward(seq)
        for out in outputs:
            assert len(out) == OUTPUT_SIZE

    def test_forward_resets_state_at_start(self):
        """Two calls to forward on the same sequence should yield identical results."""
        model = make_model()
        seq, _ = make_sequence(length=3)
        out1 = model.forward(seq)
        out2 = model.forward(seq)
        for o1, o2 in zip(out1, out2):
            for v1, v2 in zip(o1, o2):
                assert abs(v1 - v2) < 1e-12, "Repeated forward should be deterministic"

    def test_forward_state_nonzero_after_sequence(self):
        model = make_model()
        seq, _ = make_sequence(length=5)
        model.forward(seq)
        assert any(abs(s) > 1e-10 for s in model.state)


# ---------------------------------------------------------------------------
# Test: reset
# ---------------------------------------------------------------------------

class TestMambaReset:
    def test_reset_zeros_state(self):
        model = make_model()
        seq, _ = make_sequence(length=5)
        model.forward(seq)
        model.reset()
        assert all(s == 0.0 for s in model.state), \
            f"State should be zeros after reset, got {model.state[:4]}"

    def test_reset_allows_fresh_forward(self):
        """After reset, forward should give same result as a fresh model."""
        model = make_model()
        seq, _ = make_sequence(length=3)
        model.forward(seq)  # dirty state
        model.reset()
        out_reset = model.forward(seq)

        fresh = make_model()
        out_fresh = fresh.forward(seq)

        for o1, o2 in zip(out_reset, out_fresh):
            for v1, v2 in zip(o1, o2):
                assert abs(v1 - v2) < 1e-12


# ---------------------------------------------------------------------------
# Test: training
# ---------------------------------------------------------------------------

class TestMambaTraining:
    def test_train_step_returns_float(self):
        model = make_model()
        seq, targets = make_sequence(length=3)
        loss = model.train_step(seq, targets, lr=0.01)
        assert isinstance(loss, float)
        assert loss > 0

    def test_training_reduces_loss_over_epochs(self):
        model = make_model(seed=0)
        seq, targets = make_sequence(length=4, seed=1)
        losses = []
        for _ in range(8):
            loss = model.train_step(seq, targets, lr=0.05)
            losses.append(loss)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: predict
# ---------------------------------------------------------------------------

class TestMambaPredict:
    def test_predict_returns_list(self):
        model = make_model()
        seq, _ = make_sequence(length=4)
        preds = model.predict(seq)
        assert isinstance(preds, list)
        assert len(preds) == 4

    def test_predict_values_are_valid_class_indices(self):
        model = make_model()
        seq, _ = make_sequence(length=6)
        preds = model.predict(seq)
        for p in preds:
            assert isinstance(p, int)
            assert 0 <= p < OUTPUT_SIZE

    def test_predict_is_deterministic(self):
        model = make_model()
        seq, _ = make_sequence(length=4)
        preds1 = model.predict(seq)
        preds2 = model.predict(seq)
        assert preds1 == preds2


# ---------------------------------------------------------------------------
# Test: parameter counting
# ---------------------------------------------------------------------------

class TestMambaParams:
    def test_params_count_positive(self):
        model = make_model()
        assert model.get_params_count() > 0

    def test_params_count_matches_expected(self):
        """Verify the formula: A + B + C + W_select + b_select + W_out + b_out."""
        model = make_model()
        expected = (
            STATE_DIM * STATE_DIM +           # A
            STATE_DIM * INPUT_SIZE +          # B
            HIDDEN_SIZE * STATE_DIM +         # C
            STATE_DIM * INPUT_SIZE +          # W_select
            STATE_DIM +                       # b_select
            OUTPUT_SIZE * HIDDEN_SIZE +       # W_out
            OUTPUT_SIZE                       # b_out
        )
        assert model.get_params_count() == expected

    def test_larger_state_dim_has_more_params(self):
        small = MambaModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, state_dim=8, seed=0)
        large = MambaModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, state_dim=32, seed=0)
        assert large.get_params_count() > small.get_params_count()
