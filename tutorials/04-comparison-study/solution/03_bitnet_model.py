"""
Chapter 4 — BitNet: Binary/Ternary Weight Transformer
======================================================

BitNet replaces full-precision (float32) weights with ternary values {-1, 0, +1},
reducing memory by ~20x while preserving most model quality.  The key ideas:

  1. Ternary quantization — each weight snaps to one of three values based on
     its magnitude relative to a threshold.

  2. Ternary matrix multiplication — instead of multiply-accumulate, we just
     add or subtract, then scale by a learned magnitude factor.

  3. Shadow weights — training keeps full-precision "shadow" weights that
     accumulate gradient updates.  After each update, we re-quantize.
     The Straight-Through Estimator (STE) lets gradients flow through the
     quantization step as if it were the identity function.

  4. Compressed size — each ternary weight needs only ~1.58 bits vs 32 bits
     for float32, a ~20x compression ratio.

Builds on: Level A math_foundations, single_neuron, loss_function
"""

import importlib
import sys
import os
import random
import math

# ---------------------------------------------------------------------------
# Level-A imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product
vector_add = math_fn.vector_add

neuron_mod = importlib.import_module('02_single_neuron')
relu = neuron_mod.relu
sigmoid = neuron_mod.sigmoid

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss


# ---------------------------------------------------------------------------
# Helper: seeded weight initialisation
# ---------------------------------------------------------------------------

def _randn(seed, rows, cols, scale=0.1):
    """Return a rows×cols matrix of small random floats using a simple LCG."""
    rng = random.Random(seed)
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------------------
# Ternary quantization
# ---------------------------------------------------------------------------

def quantize_ternary(weights, threshold=0.5):
    """Quantize a weight matrix to {-1, 0, +1}.

    Rules for each weight w:
        |w| < threshold  →  0
        w  >  0          →  +1
        w  <  0          →  -1

    The scale factor is the mean absolute value of the weights that *survive*
    quantization (i.e. those with |w| >= threshold).  It is used during
    ternary matrix multiplication to restore the correct magnitude.

    Parameters
    ----------
    weights   : list[list[float]] — 2-D weight matrix (rows × cols)
    threshold : float — magnitude below which weights are zeroed (default 0.5)

    Returns
    -------
    (quantized, scale) where
        quantized : list[list[int]]  — same shape, values in {-1, 0, +1}
        scale     : float            — mean |w| for surviving weights (≥ 1e-8)
    """
    quantized = []
    surviving = []

    for row in weights:
        q_row = []
        for w in row:
            abs_w = abs(w)
            if abs_w < threshold:
                q_row.append(0)
            elif w > 0:
                q_row.append(1)
                surviving.append(abs_w)
            else:
                q_row.append(-1)
                surviving.append(abs_w)
        quantized.append(q_row)

    scale = sum(surviving) / len(surviving) if surviving else 1e-8
    return quantized, scale


# ---------------------------------------------------------------------------
# Ternary matrix-vector multiplication
# ---------------------------------------------------------------------------

def ternary_matmul(quantized_weights, scale, input_vec):
    """Efficient ternary matrix-vector multiply.

    For each output neuron i:
        raw[i] = sum(input_vec[j] for j where q[i][j]==+1)
               - sum(input_vec[j] for j where q[i][j]==-1)
        output[i] = raw[i] * scale

    This replaces multiply-accumulate with add/subtract + one scalar multiply,
    which is the hardware efficiency win of BitNet.

    Parameters
    ----------
    quantized_weights : list[list[int]] — ternary weight matrix (out × in)
    scale             : float           — magnitude scale factor
    input_vec         : list[float]     — input vector (length == cols)

    Returns
    -------
    list[float] — output vector (length == rows)
    """
    result = []
    for row in quantized_weights:
        acc = 0.0
        for q, x in zip(row, input_vec):
            if q == 1:
                acc += x
            elif q == -1:
                acc -= x
        result.append(acc * scale)
    return result


# ---------------------------------------------------------------------------
# BitNet model
# ---------------------------------------------------------------------------

class BitNetModel:
    """Multi-layer network with ternary weights.

    Architecture
    ------------
      Layer 0  : input_size  → hidden_size
      Layers 1…(num_layers-2) : hidden_size → hidden_size  (only when num_layers > 2)
      Last layer : hidden_size → output_size

    Training uses shadow weights (full float32) updated by SGD.
    Inference uses the quantized ternary weights.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, seed=42):
        """
        Parameters
        ----------
        input_size  : int — dimensionality of each input vector
        hidden_size : int — number of hidden neurons per layer
        output_size : int — number of output classes
        num_layers  : int — total number of weight layers (≥ 2)
        seed        : int — random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Build shadow (full-precision) weights
        # Layer shapes: [input→hidden, (hidden→hidden)×..., hidden→output]
        self.shadow_weights = []
        self.biases = []
        rng = random.Random(seed)

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                in_dim, out_dim = input_size, hidden_size
            elif layer_idx == num_layers - 1:
                in_dim, out_dim = hidden_size, output_size
            else:
                in_dim, out_dim = hidden_size, hidden_size

            W = [[rng.gauss(0, 0.5) for _ in range(in_dim)] for _ in range(out_dim)]
            b = [0.0] * out_dim
            self.shadow_weights.append(W)
            self.biases.append(b)

        # Quantized weights (list of (quantized_matrix, scale) tuples)
        self.quantized_weights = []
        self.scales = []
        self._quantize_all()

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def _quantize_all(self):
        """Quantize all shadow weight matrices to ternary."""
        self.quantized_weights = []
        self.scales = []
        for W in self.shadow_weights:
            q, s = quantize_ternary(W)
            self.quantized_weights.append(q)
            self.scales.append(s)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x):
        """Forward pass for a single input vector.

        Uses quantized (ternary) weights for all layers.
        ReLU activation after each hidden layer; no activation on output.

        Parameters
        ----------
        x : list[float] — input vector of length input_size

        Returns
        -------
        list[float] — output logits of length output_size
        """
        h = x
        for layer_idx in range(self.num_layers):
            q = self.quantized_weights[layer_idx]
            s = self.scales[layer_idx]
            b = self.biases[layer_idx]
            pre = ternary_matmul(q, s, h)
            pre_b = vector_add(pre, b)
            if layer_idx < self.num_layers - 1:
                h = [relu(z) for z in pre_b]
            else:
                h = pre_b
        return h

    def forward_sequence(self, sequence):
        """Apply forward independently to each position in the sequence.

        This is a per-position model — no recurrence between steps.

        Parameters
        ----------
        sequence : list[list[float]] — T input vectors, each of length input_size

        Returns
        -------
        list[list[float]] — T output logit vectors, each of length output_size
        """
        return [self.forward(x) for x in sequence]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, input_seq, targets, lr):
        """Single training step over a sequence.

        Uses the Straight-Through Estimator (STE): gradients flow through
        quantization as if it were the identity function — we compute
        gradients on the shadow (full-precision) weights.

        Parameters
        ----------
        input_seq : list[list[float]] — T input vectors
        targets   : list[int]         — T target class indices
        lr        : float             — learning rate

        Returns
        -------
        float — mean cross-entropy loss over the sequence
        """
        T = len(input_seq)
        total_loss = 0.0

        # Accumulate gradients over the sequence
        grad_W = [[[ 0.0] * len(self.shadow_weights[l][0])
                   for _ in range(len(self.shadow_weights[l]))]
                  for l in range(self.num_layers)]
        grad_b = [[0.0] * len(self.biases[l]) for l in range(self.num_layers)]

        for t in range(T):
            x = input_seq[t]
            target_idx = targets[t]

            # ---- Forward pass (with cached activations) ----
            activations = [x]   # activations[0] = input
            pre_acts = []       # pre-activation values per layer

            h = x
            for layer_idx in range(self.num_layers):
                q = self.quantized_weights[layer_idx]
                s = self.scales[layer_idx]
                b = self.biases[layer_idx]
                pre = ternary_matmul(q, s, h)
                pre_b = vector_add(pre, b)
                pre_acts.append(pre_b)
                if layer_idx < self.num_layers - 1:
                    h = [relu(z) for z in pre_b]
                else:
                    h = pre_b
                activations.append(h)

            logits = activations[-1]
            probs = softmax(logits)
            # Build one-hot target and compute cross-entropy
            one_hot = [0.0] * len(probs)
            one_hot[target_idx] = 1.0
            total_loss += cross_entropy_loss(probs, one_hot)

            # ---- Backward pass (STE) ----
            # Gradient of cross-entropy + softmax: dL/dlogit = prob - one_hot
            delta = list(probs)
            delta[target_idx] -= 1.0

            for layer_idx in reversed(range(self.num_layers)):
                a_in = activations[layer_idx]      # input to this layer
                pre_b = pre_acts[layer_idx]

                # Accumulate weight gradients: outer product of delta and a_in
                for i in range(len(delta)):
                    for j in range(len(a_in)):
                        grad_W[layer_idx][i][j] += delta[i] * a_in[j]
                    grad_b[layer_idx][i] += delta[i]

                if layer_idx > 0:
                    # Backprop through ReLU (STE for quantization)
                    W = self.shadow_weights[layer_idx]
                    new_delta = [0.0] * len(a_in)
                    for j in range(len(a_in)):
                        for i in range(len(delta)):
                            new_delta[j] += W[i][j] * delta[i]
                    # Apply ReLU derivative (straight-through for quantization)
                    prev_pre = pre_acts[layer_idx - 1]
                    new_delta = [new_delta[j] * (1.0 if prev_pre[j] > 0 else 0.0)
                                 for j in range(len(new_delta))]
                    delta = new_delta

        # ---- SGD update on shadow weights ----
        for layer_idx in range(self.num_layers):
            W = self.shadow_weights[layer_idx]
            for i in range(len(W)):
                for j in range(len(W[i])):
                    W[i][j] -= lr * grad_W[layer_idx][i][j] / T
                self.biases[layer_idx][i] -= lr * grad_b[layer_idx][i] / T

        # ---- Re-quantize after update ----
        self._quantize_all()

        return total_loss / T

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_seq):
        """Predict class index per position.

        Parameters
        ----------
        input_seq : list[list[float]]

        Returns
        -------
        list[int] — argmax of softmax(logits) per position
        """
        predictions = []
        for x in input_seq:
            logits = self.forward(x)
            probs = softmax(logits)
            predictions.append(probs.index(max(probs)))
        return predictions

    # ------------------------------------------------------------------
    # Size / parameter counting
    # ------------------------------------------------------------------

    def get_params_count(self):
        """Total number of parameters (weights + biases)."""
        total = 0
        for layer_idx in range(self.num_layers):
            W = self.shadow_weights[layer_idx]
            total += len(W) * len(W[0])          # weight matrix
            total += len(self.biases[layer_idx])  # bias vector
        return total

    def get_compressed_size_bytes(self):
        """Estimated storage for the quantized model.

        Ternary weights: ~1.58 bits per parameter (log2(3) ≈ 1.585).
        Scales: one float32 (4 bytes) per layer.
        Biases: stored as float32 (4 bytes each).

        Returns
        -------
        float — estimated bytes
        """
        total_ternary = 0
        total_bias_params = 0
        for layer_idx in range(self.num_layers):
            W = self.shadow_weights[layer_idx]
            total_ternary += len(W) * len(W[0])
            total_bias_params += len(self.biases[layer_idx])

        ternary_bytes = total_ternary * 1.58 / 8
        scales_bytes = self.num_layers * 4          # one float32 per layer
        bias_bytes = total_bias_params * 4           # float32 per bias
        return ternary_bytes + scales_bytes + bias_bytes


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4 — BitNet Demo")
    print("=" * 60)

    # Small demo: 4-class classifier
    INPUT_SIZE = 8
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4
    SEQ_LEN = 5

    model = BitNetModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_layers=2, seed=0)

    rng = random.Random(7)
    seq = [[rng.gauss(0, 1) for _ in range(INPUT_SIZE)] for _ in range(SEQ_LEN)]
    targets = [rng.randint(0, OUTPUT_SIZE - 1) for _ in range(SEQ_LEN)]

    print(f"\nModel parameters : {model.get_params_count()}")
    print(f"Float32 size     : {model.get_params_count() * 4:.0f} bytes")
    print(f"Compressed size  : {model.get_compressed_size_bytes():.1f} bytes")

    # Quantization check
    q, s = quantize_ternary([[0.8, -0.6, 0.1, -0.9]])
    print(f"\nquantize_ternary([[0.8,-0.6,0.1,-0.9]]) → {q}  scale={s:.4f}")

    # Training loop
    print("\n--- Training ---")
    for step in range(10):
        loss = model.train_step(seq, targets, lr=0.01)
        print(f"  step {step+1:2d}  loss={loss:.4f}")

    preds = model.predict(seq)
    print(f"\nPredictions : {preds}")
    print(f"Targets     : {targets}")
    print("=" * 60)
