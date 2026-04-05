"""
Chapter 9 — Training the Transformer
======================================

What this module teaches
-------------------------
We can now build and run a complete Transformer (Chapter 8), but its weights
are random — it produces gibberish. This chapter makes the model *learn* by
training it on data using teacher forcing.

Teacher forcing
----------------
During training, we feed the model a sequence of tokens and ask it to predict
the *next* token at every position. The input is [t0, t1, ..., tn-1] and the
target is [t1, t2, ..., tn]. At each position the model sees only the past
(enforced by the causal mask), and the loss measures how well it predicts
the known future.

Why we train only the output layer analytically
-------------------------------------------------
Full backpropagation through a Transformer requires computing gradients through
attention (softmax + masking), layer normalisation, residual connections, and
embeddings — hundreds of interacting partial derivatives. That is doable but
enormously complex in pure Python without autograd.

Instead we use a practical educational shortcut:

  1. The randomly-initialised Transformer blocks act as a **feature extractor**.
     They map token sequences to hidden-state vectors that already carry some
     structure (due to attention and positional encoding).

  2. We train the **output projection layer** (W_out, b_out) analytically.
     The gradient of cross-entropy + softmax w.r.t. the logits is simply:

         d_logit[v] = softmax(logits)[v] - one_hot_target[v]

     And the gradient w.r.t. W_out[v][j] at position t is:

         dW_out[v][j] = d_logit[v] * hidden_state[t][j]

     This is the same approach as Level A's backpropagation (Chapter 5) —
     just applied to the output layer of a Transformer.

  3. We also update the **embedding vectors** for the tokens that appear in
     each training example, using the gradient that flows back through the
     output projection.

This is enough to demonstrate real learning: the loss decreases, and the
model starts predicting the correct next tokens for patterns it has seen.

Builds on
----------
  - level-c-reader/08_stacking_layers.py  (Transformer)
  - level-a-abcs/04_loss_function.py      (softmax, cross_entropy_loss)
  - level-c-reader/01_embeddings.py       (Vocabulary)
"""

import importlib
import sys
import os
import math
import random

# ---------------------------------------------------------------------------
# Import from previous chapters
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '01-mlp-from-scratch', 'solution'))
sys.path.insert(0, os.path.dirname(__file__))

transformer_mod = importlib.import_module('08_stacking_layers')
Transformer = transformer_mod.Transformer

loss_mod = importlib.import_module('04_loss_function')
softmax = loss_mod.softmax
cross_entropy_loss = loss_mod.cross_entropy_loss

embed_mod = importlib.import_module('01_embeddings')
Vocabulary = embed_mod.Vocabulary

# We also need these for recomputing hidden states
pe_mod = importlib.import_module('02_positional_encoding')
add_position_info = pe_mod.add_position_info

attn_mod = importlib.import_module('03_self_attention')
create_causal_mask = attn_mod.create_causal_mask

math_fn = importlib.import_module('01_math_foundations')
dot_product = math_fn.dot_product


# ---------------------------------------------------------------------------
# Helper: prepare language-modelling data
# ---------------------------------------------------------------------------

def prepare_lm_data(token_indices):
    """Split a token sequence into (input, target) for next-token prediction.

    Given [t0, t1, t2, ..., tn], returns:
        input  = [t0, t1, ..., tn-1]
        target = [t1, t2, ..., tn]

    Parameters
    ----------
    token_indices : list[int] — a sequence of token IDs

    Returns
    -------
    tuple[list[int], list[int]] — (input_tokens, target_tokens)
    """
    return token_indices[:-1], token_indices[1:]


# ---------------------------------------------------------------------------
# Transformer Trainer
# ---------------------------------------------------------------------------

class TransformerTrainer:
    """Trains a Transformer's output projection layer analytically.

    The Transformer blocks act as a fixed (randomly initialised) feature
    extractor. We train W_out and b_out using exact gradients derived from
    the cross-entropy + softmax loss. We also update embedding vectors for
    tokens seen during training.

    Parameters
    ----------
    transformer : Transformer — the model to train (from Chapter 8)
    lr          : float       — learning rate (default 0.01)
    """

    def __init__(self, transformer, lr=0.01):
        self.model = transformer
        self.lr = lr

    def _get_hidden_states(self, token_indices):
        """Run the Transformer's feature extraction pipeline and return
        the hidden states *before* the output projection.

        This replicates steps 1-4 of Transformer.forward() so we can
        access the intermediate hidden-state vectors for gradient computation.

        Parameters
        ----------
        token_indices : list[int] — input token IDs

        Returns
        -------
        list[list[float]] — shape [seq_len x embed_dim], the hidden states
        """
        model = self.model
        seq_len = len(token_indices)

        # Step 1 + 2: Embeddings + positional encoding
        embeddings = model.embedding.forward(token_indices)
        x = add_position_info(embeddings, model.pe[:seq_len])

        # Step 3: Pass through all Transformer blocks with causal mask
        mask = create_causal_mask(seq_len)
        for block in model.blocks:
            x = block.forward(x, mask)

        # Step 4: Final layer norm
        x = model.final_norm.forward_sequence(x)

        return x

    def _compute_logits(self, hidden_states):
        """Project hidden states to vocabulary logits using W_out and b_out.

        Parameters
        ----------
        hidden_states : list[list[float]] — shape [seq_len x embed_dim]

        Returns
        -------
        list[list[float]] — shape [seq_len x vocab_size]
        """
        model = self.model
        logits = []
        for h in hidden_states:
            pos_logits = [
                dot_product(model.W_out[v], h) + model.b_out[v]
                for v in range(model.vocab_size)
            ]
            logits.append(pos_logits)
        return logits

    def compute_loss(self, token_indices, target_indices):
        """Compute the average cross-entropy loss across all positions.

        Parameters
        ----------
        token_indices  : list[int] — input token IDs
        target_indices : list[int] — target token IDs (same length)

        Returns
        -------
        float — average cross-entropy loss (positive scalar)
        """
        logits = self.model.forward(token_indices)
        seq_len = len(token_indices)
        total_loss = 0.0

        for t in range(seq_len):
            probs = softmax(logits[t])
            # Build one-hot target
            one_hot = [0.0] * self.model.vocab_size
            one_hot[target_indices[t]] = 1.0
            total_loss += cross_entropy_loss(probs, one_hot)

        return total_loss / seq_len

    def train_step(self, token_indices, target_indices):
        """Perform one training step: forward pass, compute loss, update weights.

        Uses the analytical gradient of cross-entropy + softmax:
            d_logit[v] = softmax(logits)[v] - one_hot[v]
            dW_out[v][j] = d_logit[v] * hidden[t][j]
            db_out[v] = d_logit[v]

        Parameters
        ----------
        token_indices  : list[int] — input token IDs
        target_indices : list[int] — target token IDs (same length)

        Returns
        -------
        float — the loss *before* the weight update (for monitoring)
        """
        model = self.model
        seq_len = len(token_indices)
        vocab_size = model.vocab_size
        embed_dim = model.embed_dim

        # Forward pass: get hidden states and logits
        hidden_states = self._get_hidden_states(token_indices)
        logits = self._compute_logits(hidden_states)

        # Compute loss and gradients
        total_loss = 0.0

        # Accumulate gradients for W_out and b_out
        # We only accumulate for vocab entries that are either targets or
        # have significant probability mass, for efficiency.
        dW_out = [[0.0] * embed_dim for _ in range(vocab_size)]
        db_out = [0.0] * vocab_size

        # Also accumulate gradients for embeddings (backprop through output layer)
        # d_hidden[t][j] = sum_v (d_logit[v] * W_out[v][j])
        d_hidden = [[0.0] * embed_dim for _ in range(seq_len)]

        for t in range(seq_len):
            probs = softmax(logits[t])

            # One-hot target
            one_hot = [0.0] * vocab_size
            one_hot[target_indices[t]] = 1.0

            # Loss for this position
            total_loss += cross_entropy_loss(probs, one_hot)

            # Gradient of loss w.r.t. logits: (softmax - one_hot)
            d_logit = [probs[v] - one_hot[v] for v in range(vocab_size)]

            # Accumulate W_out and b_out gradients
            for v in range(vocab_size):
                if abs(d_logit[v]) < 1e-8:
                    continue  # skip near-zero gradients for efficiency
                for j in range(embed_dim):
                    dW_out[v][j] += d_logit[v] * hidden_states[t][j]
                db_out[v] += d_logit[v]

            # Gradient w.r.t. hidden states (for embedding update)
            for j in range(embed_dim):
                for v in range(vocab_size):
                    if abs(d_logit[v]) < 1e-8:
                        continue
                    d_hidden[t][j] += d_logit[v] * model.W_out[v][j]

        avg_loss = total_loss / seq_len

        # --- Update output projection weights ---
        scale = self.lr / seq_len
        for v in range(vocab_size):
            for j in range(embed_dim):
                model.W_out[v][j] -= scale * dW_out[v][j]
            model.b_out[v] -= scale * db_out[v]

        # --- Update embedding vectors for tokens in this example ---
        # The gradient flows: loss → logits → hidden → ... → embedding
        # Since the transformer blocks are fixed, the gradient through them
        # is complex. But we can directly update embeddings using d_hidden
        # as an approximate signal (treating the transformer as linear locally).
        for t in range(seq_len):
            token_id = token_indices[t]
            for j in range(embed_dim):
                model.embedding.weights[token_id][j] -= scale * d_hidden[t][j]

        return avg_loss

    def train(self, dataset, epochs, verbose=True):
        """Train the model on a dataset for multiple epochs.

        Parameters
        ----------
        dataset : list[tuple[list[int], list[int]]]
            Each element is (input_tokens, target_tokens).
        epochs  : int — number of passes through the full dataset.
        verbose : bool — if True, print loss after each epoch.

        Returns
        -------
        list[float] — average loss per epoch
        """
        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            for input_tokens, target_tokens in dataset:
                loss = self.train_step(input_tokens, target_tokens)
                total_loss += loss

            avg_loss = total_loss / len(dataset)
            epoch_losses.append(avg_loss)

            if verbose:
                print(f"  Epoch {epoch + 1:3d}/{epochs}  loss = {avg_loss:.4f}")

        return epoch_losses


# ---------------------------------------------------------------------------
# Demo — run with:
#   python3 "phase1-from-scratch/level-c-reader/09_training.py"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 9 — Training the Transformer")
    print("Teacher forcing with analytical output-layer gradients")
    print("=" * 60)

    # Build a tiny vocabulary for a repeating pattern
    vocab = Vocabulary()
    vocab.build([
        "a b c d a b c d",
        "the cat sat the cat sat",
    ])
    vocab_size = vocab.size
    print(f"\nVocabulary size: {vocab_size}")

    # Create a small Transformer
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=32,
        seed=42,
    )
    print(f"Model: embed_dim=16, heads=2, layers=1")
    print(f"Total parameters: {model.get_params_count():,}")

    # Prepare training data: a repeating pattern
    # Encode "a b c d a b c d" and split into input/target
    sentence = "a b c d a b c d"
    token_ids = vocab.encode_sentence(sentence)
    print(f"\nTraining sentence: '{sentence}'")
    print(f"Token IDs: {token_ids}")

    input_seq, target_seq = prepare_lm_data(token_ids)
    print(f"Input:  {input_seq}")
    print(f"Target: {target_seq}")

    dataset = [(input_seq, target_seq)]

    # Train
    trainer = TransformerTrainer(model, lr=0.05)

    print(f"\nTraining for 20 epochs...")
    losses = trainer.train(dataset, epochs=20, verbose=True)

    print(f"\nLoss reduction: {losses[0]:.4f} → {losses[-1]:.4f}")
    if losses[-1] < losses[0]:
        print("The model is learning! Loss decreased.")
    else:
        print("Loss did not decrease (unusual for this setup).")

    # Show predictions after training
    print("\nPredictions after training:")
    logits = model.forward(input_seq)
    for i, pos_logits in enumerate(logits):
        probs = softmax(pos_logits)
        best_idx = max(range(vocab_size), key=lambda v: probs[v])
        best_word = vocab.decode(best_idx)
        target_word = vocab.decode(target_seq[i])
        correct = "Y" if best_idx == target_seq[i] else " "
        print(f"  pos {i}: predicted '{best_word}' "
              f"(p={probs[best_idx]:.3f}), "
              f"target '{target_word}' [{correct}]")

    print("\n" + "=" * 60)
    print("Chapter 9 complete. The Transformer can learn from data.")
    print("=" * 60)
