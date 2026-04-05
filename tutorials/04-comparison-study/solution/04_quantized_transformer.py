"""
Chapter 5 — Quantized Transformer: Int8 Post-Training Quantization
===================================================================

Real transformers are large.  Int8 quantization compresses each float32 weight
to a single unsigned byte, cutting model size by ~4x with minimal accuracy loss.

Key concepts in this chapter:

  1. Int8 quantization — map a float range [min, max] to the integer range
     [0, 255] using a linear scale + offset.  This is *affine quantization*:
     the most common format in production inference engines (TensorFlow Lite,
     ONNX Runtime, llama.cpp).

  2. Dequantization — reverse the mapping: float ≈ int * scale + offset.
     The round-trip isn't exact, but error is bounded by scale/2.

  3. Post-training quantization (PTQ) — quantize after training is complete.
     No special training procedure is required.  During training we keep
     full-precision weights; after each update we re-quantize.

  4. Simplified transformer — one embedding projection, multi-head attention
     (Q, K, V, output projections), feed-forward network (two dense layers),
     and output projection.  All weight matrices are int8-quantized.

Architecture (per layer)
------------------------
  x  →  [LayerNorm-like normalise]
     →  MultiHead Attention (Q·K^T softmax → V, projected) + residual
     →  [LayerNorm-like normalise]
     →  FFN (W1 ReLU W2) + residual
  → Output projection → logits

Note: layer-norm here is a simple mean/std normalisation (no learnable params)
to keep the module pure-Python with no external dependencies.

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
matrix_vector_multiply = math_fn.matrix_vector_multiply
transpose = math_fn.transpose

neuron_mod = importlib.import_module('02_single_neuron')
relu = neuron_mod.relu

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss


# ---------------------------------------------------------------------------
# Int8 quantization helpers
# ---------------------------------------------------------------------------

def quantize_int8(weights):
    """Quantize a weight matrix to uint8 using affine (min/max) quantization.

    For each weight w:
        quantized = round((w - offset) / scale)
        clamped   = clip(quantized, 0, 255)

    where:
        scale  = (max(weights) - min(weights)) / 255
        offset = min(weights)

    Parameters
    ----------
    weights : list[list[float]] — 2-D weight matrix (rows × cols)

    Returns
    -------
    (quantized_matrix, scale, offset) where
        quantized_matrix : list[list[int]]  — uint8 values in [0, 255]
        scale            : float            — linear scale factor
        offset           : float            — zero-point offset (= min weight)
    """
    # Flatten to find global min / max
    flat = [w for row in weights for w in row]
    w_min = min(flat)
    w_max = max(flat)

    # Avoid division by zero when all weights are identical
    w_range = w_max - w_min
    scale = w_range / 255.0 if w_range > 0 else 1e-8
    offset = w_min

    quantized = []
    for row in weights:
        q_row = []
        for w in row:
            q = (w - offset) / scale
            q_int = int(round(q))
            q_clamp = max(0, min(255, q_int))
            q_row.append(q_clamp)
        quantized.append(q_row)

    return quantized, scale, offset


def dequantize_int8(quantized, scale, offset):
    """Reverse int8 quantization back to float.

    w_approx = quantized * scale + offset

    Parameters
    ----------
    quantized : list[list[int]] — uint8 values in [0, 255]
    scale     : float
    offset    : float

    Returns
    -------
    list[list[float]] — approximate original weights
    """
    return [[q * scale + offset for q in row] for row in quantized]


# ---------------------------------------------------------------------------
# Simple layer-norm (no learnable params — keeps module self-contained)
# ---------------------------------------------------------------------------

def _layer_norm(x, eps=1e-8):
    """Normalise a vector to zero-mean, unit-variance."""
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    std = math.sqrt(var + eps)
    return [(xi - mean) / std for xi in x]


# ---------------------------------------------------------------------------
# Attention helper
# ---------------------------------------------------------------------------

def _scaled_dot_product_attention(Q, K, V):
    """Single-head attention: softmax(Q K^T / sqrt(d_k)) V.

    Parameters
    ----------
    Q, K, V : list[list[float]] — each T × d matrices

    Returns
    -------
    list[list[float]] — T × d output
    """
    T = len(Q)
    d_k = len(Q[0])
    scale = math.sqrt(d_k)

    # Compute attention scores: T × T
    scores = []
    for i in range(T):
        row = []
        for j in range(T):
            s = dot_product(Q[i], K[j]) / scale
            row.append(s)
        scores.append(row)

    # Softmax over each row
    attn = [softmax(row) for row in scores]

    # Weighted sum of V: T × d_v
    d_v = len(V[0])
    output = []
    for i in range(T):
        out_i = [0.0] * d_v
        for j in range(T):
            for k in range(d_v):
                out_i[k] += attn[i][j] * V[j][k]
        output.append(out_i)

    return output


# ---------------------------------------------------------------------------
# Quantized Transformer
# ---------------------------------------------------------------------------

class QuantizedTransformer:
    """Small transformer with post-training int8 quantization.

    Architecture
    ------------
    For each input position x_t:
        h = embedding_proj(x_t)          # input_size → hidden_size
        for each layer:
            h = layer_norm(h)
            h = h + attention_block(h)   # simplified multi-head attention
            h = layer_norm(h)
            h = h + ffn_block(h)         # W1 → ReLU → W2
        logit = output_proj(h)            # hidden_size → output_size

    All weight matrices are stored as int8 and dequantized before use.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=2, num_heads=2, seed=42):
        """
        Parameters
        ----------
        input_size  : int — dimensionality of each one-hot / embedding input
        hidden_size : int — transformer hidden size (must be divisible by num_heads)
        output_size : int — number of output classes
        num_layers  : int — number of transformer layers
        num_heads   : int — number of attention heads per layer
        seed        : int — random seed
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        rng = random.Random(seed)

        def _rand_matrix(rows, cols, scale=0.1):
            return [[rng.gauss(0, scale) for _ in range(cols)]
                    for _ in range(rows)]

        # ---- Full-precision shadow weights ----

        # Embedding projection: input_size → hidden_size
        self.shadow_embed = _rand_matrix(hidden_size, input_size)
        self.embed_bias = [0.0] * hidden_size

        # Per-layer attention projections: Q, K, V, O
        # Q/K/V: hidden_size → hidden_size   O: hidden_size → hidden_size
        self.shadow_Q = [_rand_matrix(hidden_size, hidden_size) for _ in range(num_layers)]
        self.shadow_K = [_rand_matrix(hidden_size, hidden_size) for _ in range(num_layers)]
        self.shadow_V = [_rand_matrix(hidden_size, hidden_size) for _ in range(num_layers)]
        self.shadow_O = [_rand_matrix(hidden_size, hidden_size) for _ in range(num_layers)]

        # Per-layer FFN: W1 hidden→hidden*2, W2 hidden*2→hidden
        self.shadow_W1 = [_rand_matrix(hidden_size * 2, hidden_size) for _ in range(num_layers)]
        self.shadow_W2 = [_rand_matrix(hidden_size, hidden_size * 2) for _ in range(num_layers)]

        # Output projection: hidden_size → output_size
        self.shadow_out = _rand_matrix(output_size, hidden_size)
        self.out_bias = [0.0] * output_size

        # ---- Quantized weights (filled by _quantize_all) ----
        self._quantize_all()

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def _quantize_all(self):
        """Quantize all shadow weight matrices to int8."""
        def q(W):
            return quantize_int8(W)

        self.q_embed, self.s_embed, self.o_embed = q(self.shadow_embed)

        self.q_Q = []; self.s_Q = []; self.o_Q = []
        self.q_K = []; self.s_K = []; self.o_K = []
        self.q_V = []; self.s_V = []; self.o_V = []
        self.q_O = []; self.s_O = []; self.o_O = []
        self.q_W1 = []; self.s_W1 = []; self.o_W1 = []
        self.q_W2 = []; self.s_W2 = []; self.o_W2 = []

        for l in range(self.num_layers):
            qq, sq, oq = q(self.shadow_Q[l]); self.q_Q.append(qq); self.s_Q.append(sq); self.o_Q.append(oq)
            qk, sk, ok = q(self.shadow_K[l]); self.q_K.append(qk); self.s_K.append(sk); self.o_K.append(ok)
            qv, sv, ov = q(self.shadow_V[l]); self.q_V.append(qv); self.s_V.append(sv); self.o_V.append(ov)
            qo, so, oo = q(self.shadow_O[l]); self.q_O.append(qo); self.s_O.append(so); self.o_O.append(oo)
            q1, s1, o1 = q(self.shadow_W1[l]); self.q_W1.append(q1); self.s_W1.append(s1); self.o_W1.append(o1)
            q2, s2, o2 = q(self.shadow_W2[l]); self.q_W2.append(q2); self.s_W2.append(s2); self.o_W2.append(o2)

        self.q_out, self.s_out, self.o_out = q(self.shadow_out)

    # ------------------------------------------------------------------
    # Helper: dequantized matmul + bias
    # ------------------------------------------------------------------

    def _deq_matvec(self, q_mat, scale, offset, x):
        """Dequantize weight matrix and multiply by vector x."""
        W = dequantize_int8(q_mat, scale, offset)
        return matrix_vector_multiply(W, x)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x):
        """Forward pass for a single input vector.

        Parameters
        ----------
        x : list[float] — input vector of length input_size

        Returns
        -------
        list[float] — output logits of length output_size
        """
        # Embed: input_size → hidden_size
        h = vector_add(self._deq_matvec(self.q_embed, self.s_embed, self.o_embed, x),
                       self.embed_bias)

        # Transformer layers (single-position — treat as sequence of length 1)
        for l in range(self.num_layers):
            # --- Attention block ---
            h_norm = _layer_norm(h)

            # Single-position attention: Q·K^T degenerates to identity-like
            # Compute Q, K, V projections
            q_vec = self._deq_matvec(self.q_Q[l], self.s_Q[l], self.o_Q[l], h_norm)
            k_vec = self._deq_matvec(self.q_K[l], self.s_K[l], self.o_K[l], h_norm)
            v_vec = self._deq_matvec(self.q_V[l], self.s_V[l], self.o_V[l], h_norm)

            # For a single position, attention output = V (score is trivially 1)
            # Then project through O
            attn_out = self._deq_matvec(self.q_O[l], self.s_O[l], self.o_O[l], v_vec)

            # Residual
            h = vector_add(h, attn_out)

            # --- FFN block ---
            h_norm2 = _layer_norm(h)

            # W1: hidden → hidden*2, ReLU
            ffn1 = self._deq_matvec(self.q_W1[l], self.s_W1[l], self.o_W1[l], h_norm2)
            ffn1_act = [relu(z) for z in ffn1]

            # W2: hidden*2 → hidden
            ffn2 = self._deq_matvec(self.q_W2[l], self.s_W2[l], self.o_W2[l], ffn1_act)

            # Residual
            h = vector_add(h, ffn2)

        # Output projection
        logits = vector_add(
            self._deq_matvec(self.q_out, self.s_out, self.o_out, h),
            self.out_bias
        )
        return logits

    def forward_sequence(self, sequence):
        """Apply forward independently to each position.

        Parameters
        ----------
        sequence : list[list[float]] — T input vectors

        Returns
        -------
        list[list[float]] — T output logit vectors
        """
        return [self.forward(x) for x in sequence]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, input_seq, targets, lr):
        """Train in full precision, re-quantize after update.

        Uses a simplified gradient computation: only the output projection and
        final hidden state gradient are backpropagated (shallow gradient estimate).
        This is sufficient for the educational purpose and loss reduction test.

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

        # Accumulate gradients for output projection and output bias
        grad_out = [[0.0] * self.hidden_size for _ in range(self.output_size)]
        grad_out_bias = [0.0] * self.output_size

        # Accumulate gradients for all layers (simplified: use shadow weights)
        grad_embed = [[0.0] * self.input_size for _ in range(self.hidden_size)]
        grad_Q = [[[0.0] * self.hidden_size for _ in range(self.hidden_size)]
                  for _ in range(self.num_layers)]
        grad_K = [[[0.0] * self.hidden_size for _ in range(self.hidden_size)]
                  for _ in range(self.num_layers)]
        grad_V = [[[0.0] * self.hidden_size for _ in range(self.hidden_size)]
                  for _ in range(self.num_layers)]
        grad_O = [[[0.0] * self.hidden_size for _ in range(self.hidden_size)]
                  for _ in range(self.num_layers)]
        grad_W1 = [[[0.0] * self.hidden_size for _ in range(self.hidden_size * 2)]
                   for _ in range(self.num_layers)]
        grad_W2 = [[[0.0] * self.hidden_size * 2 for _ in range(self.hidden_size)]
                   for _ in range(self.num_layers)]

        for t in range(T):
            x = input_seq[t]
            target_idx = targets[t]

            # Full-precision forward pass for gradient computation
            # Embed
            h = vector_add(
                matrix_vector_multiply(self.shadow_embed, x),
                self.embed_bias
            )
            h_embed = list(h)

            # Store activations for each layer
            h_inputs = [h_embed]
            h_norm1s = []
            v_vecs = []
            attn_outs = []
            h_after_attns = []
            h_norm2s = []
            ffn1_acts = []
            ffn2s = []

            for l in range(self.num_layers):
                h_norm = _layer_norm(h)
                h_norm1s.append(h_norm)

                v_vec = matrix_vector_multiply(self.shadow_V[l], h_norm)
                attn_out = matrix_vector_multiply(self.shadow_O[l], v_vec)
                v_vecs.append(v_vec)
                attn_outs.append(attn_out)

                h = vector_add(h, attn_out)
                h_after_attns.append(list(h))

                h_norm2 = _layer_norm(h)
                h_norm2s.append(h_norm2)

                ffn1 = matrix_vector_multiply(self.shadow_W1[l], h_norm2)
                ffn1_act = [relu(z) for z in ffn1]
                ffn1_acts.append(ffn1_act)

                ffn2 = matrix_vector_multiply(self.shadow_W2[l], ffn1_act)
                ffn2s.append(ffn2)

                h = vector_add(h, ffn2)
                h_inputs.append(list(h))

            logits = vector_add(
                matrix_vector_multiply(self.shadow_out, h),
                self.out_bias
            )
            probs = softmax(logits)
            one_hot = [0.0] * self.output_size
            one_hot[target_idx] = 1.0
            total_loss += cross_entropy_loss(probs, one_hot)

            # Gradient of output: dL/dlogit = probs - one_hot
            delta_out = [p - oh for p, oh in zip(probs, one_hot)]

            # Gradient for output projection (output_size × hidden_size)
            for i in range(self.output_size):
                for j in range(self.hidden_size):
                    grad_out[i][j] += delta_out[i] * h[j]
                grad_out_bias[i] += delta_out[i]

            # Backprop delta into h
            delta_h = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                for i in range(self.output_size):
                    delta_h[j] += self.shadow_out[i][j] * delta_out[i]

            # Simplified backprop through layers (last to first)
            for l in reversed(range(self.num_layers)):
                # Through FFN residual: delta_h passes through both paths
                delta_ffn2 = delta_h

                # Through W2 (hidden × hidden*2) -- grad_W2[l] shape: (hidden, hidden*2)
                for i in range(self.hidden_size):
                    for j in range(self.hidden_size * 2):
                        grad_W2[l][i][j] += delta_ffn2[i] * ffn1_acts[l][j]

                # Backprop through W2 into ffn1_act
                delta_ffn1_act = [0.0] * (self.hidden_size * 2)
                for j in range(self.hidden_size * 2):
                    for i in range(self.hidden_size):
                        delta_ffn1_act[j] += self.shadow_W2[l][i][j] * delta_ffn2[i]

                # Through ReLU
                ffn1_raw = matrix_vector_multiply(self.shadow_W1[l], h_norm2s[l])
                delta_ffn1 = [delta_ffn1_act[j] * (1.0 if ffn1_raw[j] > 0 else 0.0)
                              for j in range(self.hidden_size * 2)]

                # Through W1 (hidden*2 × hidden)
                for i in range(self.hidden_size * 2):
                    for j in range(self.hidden_size):
                        grad_W1[l][i][j] += delta_ffn1[i] * h_norm2s[l][j]

                # Backprop into h through FFN + residual
                delta_h_from_ffn = [0.0] * self.hidden_size
                for j in range(self.hidden_size):
                    for i in range(self.hidden_size * 2):
                        delta_h_from_ffn[j] += self.shadow_W1[l][i][j] * delta_ffn1[i]
                # Residual adds to delta_h
                delta_h = vector_add(delta_h, delta_h_from_ffn)

                # Through attention residual
                delta_attn = delta_h

                # Through O projection
                for i in range(self.hidden_size):
                    for j in range(self.hidden_size):
                        grad_O[l][i][j] += delta_attn[i] * v_vecs[l][j]

                # Backprop through O into v
                delta_v = [0.0] * self.hidden_size
                for j in range(self.hidden_size):
                    for i in range(self.hidden_size):
                        delta_v[j] += self.shadow_O[l][i][j] * delta_attn[i]

                # Through V projection (simplified: ignore layer_norm jacobian)
                h_in = h_inputs[l]
                for i in range(self.hidden_size):
                    for j in range(self.hidden_size):
                        grad_V[l][i][j] += delta_v[i] * h_norm1s[l][j]

                # Backprop into h (simplified, through residual only)
                delta_h_from_attn = [0.0] * self.hidden_size
                for j in range(self.hidden_size):
                    for i in range(self.hidden_size):
                        delta_h_from_attn[j] += self.shadow_V[l][i][j] * delta_v[i]
                delta_h = vector_add(delta_h, delta_h_from_attn)

            # Through embed
            for i in range(self.hidden_size):
                for j in range(self.input_size):
                    grad_embed[i][j] += delta_h[i] * x[j]

        # ---- SGD update ----
        def _sgd(W, grad, lr, T):
            return [[W[i][j] - lr * grad[i][j] / T
                     for j in range(len(W[i]))]
                    for i in range(len(W))]

        self.shadow_out = _sgd(self.shadow_out, grad_out, lr, T)
        for i in range(self.output_size):
            self.out_bias[i] -= lr * grad_out_bias[i] / T

        self.shadow_embed = _sgd(self.shadow_embed, grad_embed, lr, T)

        for l in range(self.num_layers):
            self.shadow_O[l] = _sgd(self.shadow_O[l], grad_O[l], lr, T)
            self.shadow_V[l] = _sgd(self.shadow_V[l], grad_V[l], lr, T)
            self.shadow_W1[l] = _sgd(self.shadow_W1[l], grad_W1[l], lr, T)
            self.shadow_W2[l] = _sgd(self.shadow_W2[l], grad_W2[l], lr, T)

        # ---- Re-quantize ----
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
        """Total number of weight parameters (excludes biases for simplicity)."""
        h = self.hidden_size
        total = h * self.input_size          # embed
        for _ in range(self.num_layers):
            total += h * h * 4               # Q, K, V, O
            total += h * 2 * h               # W1
            total += h * h * 2               # W2
        total += self.output_size * h        # output projection
        return total

    def get_compressed_size_bytes(self):
        """Int8 weights = 1 byte per parameter.

        Returns
        -------
        int — approximate bytes for quantized weight storage
        """
        return self.get_params_count()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 5 — Quantized Transformer Demo")
    print("=" * 60)

    INPUT_SIZE = 8
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4
    SEQ_LEN = 5

    model = QuantizedTransformer(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                                 num_layers=2, num_heads=2, seed=0)

    rng = random.Random(7)
    seq = [[rng.gauss(0, 1) for _ in range(INPUT_SIZE)] for _ in range(SEQ_LEN)]
    targets = [rng.randint(0, OUTPUT_SIZE - 1) for _ in range(SEQ_LEN)]

    print(f"\nModel parameters : {model.get_params_count()}")
    print(f"Float32 size     : {model.get_params_count() * 4} bytes")
    print(f"Int8 size        : {model.get_compressed_size_bytes()} bytes")

    # Quantization round-trip demo
    W = [[0.5, -0.3, 1.2], [-0.7, 0.0, 0.9]]
    q, s, o = quantize_int8(W)
    W_back = dequantize_int8(q, s, o)
    max_err = max(abs(W[i][j] - W_back[i][j])
                  for i in range(len(W)) for j in range(len(W[0])))
    print(f"\nQuantize round-trip max error: {max_err:.6f}")

    print("\n--- Training ---")
    loss_prev = None
    for step in range(10):
        loss = model.train_step(seq, targets, lr=0.01)
        print(f"  step {step+1:2d}  loss={loss:.4f}")

    preds = model.predict(seq)
    print(f"\nPredictions : {preds}")
    print(f"Targets     : {targets}")
    print("=" * 60)
