"""
Tests for Level D — Mixture of Experts Model (02_moe_model.py)

Run with: python3 -m pytest tests/test_level_d/test_02_moe.py -v
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

moe_mod = importlib.import_module('01_moe_model')
MoEModel = moe_mod.MoEModel
Expert = moe_mod.Expert
Router = moe_mod.Router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_SIZE = 8
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4
NUM_EXPERTS = 4
TOP_K = 2


def make_model(seed=42):
    return MoEModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                    num_experts=NUM_EXPERTS, top_k=TOP_K, seed=seed)


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
# Test: model creation
# ---------------------------------------------------------------------------

class TestMoECreation:
    def test_model_instantiates(self):
        model = make_model()
        assert model is not None

    def test_correct_number_of_experts(self):
        model = make_model()
        assert len(model.experts) == NUM_EXPERTS

    def test_top_k_stored(self):
        model = make_model()
        assert model.top_k == TOP_K

    def test_router_exists(self):
        model = make_model()
        assert model.router is not None
        assert isinstance(model.router, Router)


# ---------------------------------------------------------------------------
# Test: forward pass
# ---------------------------------------------------------------------------

class TestMoEForward:
    def test_forward_returns_correct_output_size(self):
        model = make_model()
        x = one_hot(3, INPUT_SIZE)
        output, _ = model.forward(x)
        assert len(output) == OUTPUT_SIZE

    def test_forward_routing_info_has_top_k_indices(self):
        model = make_model()
        x = one_hot(0, INPUT_SIZE)
        _, (indices, weights) = model.forward(x)
        assert len(indices) == TOP_K
        assert len(weights) == TOP_K

    def test_router_selects_exactly_top_k(self):
        model = make_model()
        x = one_hot(2, INPUT_SIZE)
        _, (indices, _) = model.forward(x)
        # Indices must be distinct and within range
        assert len(set(indices)) == TOP_K
        for idx in indices:
            assert 0 <= idx < NUM_EXPERTS

    def test_routing_weights_sum_to_one(self):
        model = make_model()
        x = one_hot(1, INPUT_SIZE)
        _, (_, weights) = model.forward(x)
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_forward_sequence_returns_one_output_per_step(self):
        model = make_model()
        seq, _ = make_sequence(length=6)
        results = model.forward_sequence(seq)
        assert len(results) == 6

    def test_forward_sequence_each_step_correct_shape(self):
        model = make_model()
        seq, _ = make_sequence(length=4)
        results = model.forward_sequence(seq)
        for output, _ in results:
            assert len(output) == OUTPUT_SIZE


# ---------------------------------------------------------------------------
# Test: sparsity (active < total params)
# ---------------------------------------------------------------------------

class TestMoESparsity:
    def test_active_params_less_than_total(self):
        model = make_model()
        assert model.get_active_params_count() < model.get_params_count()

    def test_active_params_positive(self):
        model = make_model()
        assert model.get_active_params_count() > 0

    def test_total_params_positive(self):
        model = make_model()
        assert model.get_params_count() > 0

    def test_active_params_ratio_matches_top_k(self):
        """Active expert params should be top_k / num_experts of total expert params."""
        model = make_model()
        router_params = model.router.param_count
        total_expert_params = sum(e.param_count for e in model.experts)
        active_expert_params = model.get_active_params_count() - router_params
        expected_active = (TOP_K / NUM_EXPERTS) * total_expert_params
        # Allow small tolerance due to integer rounding
        assert abs(active_expert_params - expected_active) < 1


# ---------------------------------------------------------------------------
# Test: training reduces loss
# ---------------------------------------------------------------------------

class TestMoETraining:
    def test_training_reduces_loss_over_epochs(self):
        model = make_model(seed=0)
        seq, targets = make_sequence(length=4, seed=1)
        losses = []
        for _ in range(10):
            loss = model.train_step(seq, targets, lr=0.05)
            losses.append(loss)
        # Loss should be lower at end than at start
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )

    def test_train_step_returns_float(self):
        model = make_model()
        seq, targets = make_sequence(length=3)
        loss = model.train_step(seq, targets, lr=0.01)
        assert isinstance(loss, float)
        assert loss > 0


# ---------------------------------------------------------------------------
# Test: routing diversity
# ---------------------------------------------------------------------------

class TestMoERouting:
    def test_different_inputs_can_route_to_different_experts(self):
        """With enough variation in inputs, not all steps should route identically."""
        model = make_model(seed=99)
        rng = random.Random(42)
        routing_sets = set()
        for _ in range(20):
            x = one_hot(rng.randint(0, INPUT_SIZE - 1), INPUT_SIZE)
            _, (indices, _) = model.forward(x)
            routing_sets.add(tuple(sorted(indices)))
        # At least 2 distinct routing combinations should appear
        assert len(routing_sets) >= 2, (
            f"All 20 inputs routed to the same experts: {routing_sets}"
        )

    def test_same_input_always_routes_same(self):
        """Deterministic: the same input should always select the same experts."""
        model = make_model(seed=42)
        x = one_hot(5, INPUT_SIZE)
        _, (idx1, _) = model.forward(x)
        _, (idx2, _) = model.forward(x)
        assert idx1 == idx2


# ---------------------------------------------------------------------------
# Test: predict
# ---------------------------------------------------------------------------

class TestMoEPredict:
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
