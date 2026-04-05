# Lesson: MLP from Scratch

A detailed walkthrough of every chapter. Read this alongside (or before) the code.

---

## Chapter 1: Math Foundations

Neural networks are, at their core, a chain of matrix multiplications with activation functions between them. That's it. Everything else is bookkeeping.

So we start with the operations themselves.

### Vectors

A vector is just a list of numbers:

```python
weights = [0.5, -0.3, 0.8]
inputs  = [1.0,  2.0, 3.0]
```

We need four operations: add, subtract, scale, and dot product.

**Addition** (`vector_add`): element-wise. `[1,2,3] + [4,5,6] = [5,7,9]`.

Used in: adding a bias vector to a layer's weighted sum (`z = W @ x + b`).

**Subtraction** (`vector_subtract`): element-wise. `[5,7,9] - [4,5,6] = [1,2,3]`.

Used in: SGD weight updates (`w_new = w_old - learning_rate * gradient`).

**Scaling** (`vector_scale`): multiply every element by a constant.

Used in: applying the learning rate before a weight update.

**Dot product** (`dot_product`): multiply corresponding elements and sum them.

```
dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

This is the fundamental computation of a single neuron. One neuron does exactly: `z = dot(weights, inputs) + bias`.

### Matrices

A matrix is a list of vectors (rows). We need:

**Transpose**: swap rows and columns. Shape (m, n) → (n, m). Used in backpropagation.

**Matrix-vector multiply**: `result[i] = dot(matrix_row[i], vector)`. This is how a full layer transforms an input — one row per neuron.

**Matrix multiply**: general case. Used for batch processing.

The key insight: when you see `W @ x` in a paper, it's just `matrix_vector_multiply(W, x)`.

---

## Chapter 2: Single Neuron

A neuron does two things:

1. Weighted sum: `z = dot(weights, inputs) + bias`
2. Activation: `a = activation(z)`

Without activation functions, stacking neurons would only produce linear transformations. No matter how many layers you add, the whole thing collapses to a single linear function. Activations break this.

### Sigmoid

```python
sigmoid(z) = 1 / (1 + exp(-z))
```

Output is always in (0, 1). Useful for probabilities. The problem: its gradient (`a * (1 - a)`) is less than 0.25 everywhere, which means gradients shrink as they pass through sigmoid layers — the vanishing gradient problem.

### ReLU

```python
relu(z) = max(0, z)
```

Output is 0 or z. Much simpler. The gradient is either 0 or 1, so it doesn't suppress gradients the way sigmoid does. This is why modern networks use ReLU (or variants) in hidden layers.

### Implementation note

The `Neuron` class stores its inputs, `z`, and `a` after each forward pass. These stored values are *required* for backpropagation. Don't discard them.

---

## Chapter 3: Forward Pass

A layer is many neurons processing the same input in parallel. Mathematically:

```
z = W @ x + b         # W is [output_size x input_size], x is [input_size]
a = activation(z)     # applied element-wise
```

A network is layers chained together — the output of one becomes the input of the next.

### Xavier Initialization

Weights can't start at zero (all neurons learn the same thing — symmetry breaking). They can't be too large (outputs explode). Xavier initialization picks a sensible scale:

```python
limit = sqrt(6 / (fan_in + fan_out))
weights ~ Uniform(-limit, limit)
```

This keeps activations in a reasonable range regardless of layer size.

### Stored Intermediates

Every `Layer.forward()` call stores:
- `self.inputs`: what came in
- `self.z_values`: pre-activation values
- `self.outputs`: post-activation values

The backward pass reads these directly. If you call `forward()` again before `backward()`, you'll overwrite the values and get wrong gradients.

---

## Chapter 4: Loss Functions

A loss function is a single number measuring "how wrong the model is." Training is the process of making this number smaller.

### Softmax

The output layer produces raw scores ("logits"). Softmax converts them to probabilities:

```
softmax(x)_i = exp(x_i) / sum(exp(x_j) for all j)
```

All outputs are in (0, 1) and sum to 1. This lets us interpret them as probabilities.

**Numerical stability**: if any `x_i` is large, `exp(x_i)` overflows. Solution: subtract the maximum before exponentiating. This doesn't change the result (the constant cancels in the ratio) but keeps all exponents at or below 0.

### Cross-Entropy Loss

For a classification problem:

```
loss = -sum(target_i * log(predicted_i))
```

For one-hot targets this simplifies to `-log(predicted[true_class])`. If the model assigns probability 1.0 to the correct class, loss = 0. As the probability falls toward 0, loss grows toward infinity.

**Why cross-entropy instead of MSE for classification?** The gradient of cross-entropy w.r.t. the logits (when paired with softmax) simplifies beautifully to `softmax(z) - target`. MSE doesn't have this property and converges more slowly.

---

## Chapter 5: Backpropagation

This is the chapter that makes people nervous. It shouldn't.

Backpropagation is the chain rule from calculus applied to a composition of functions. The chain rule says: if `y = f(g(x))`, then `dy/dx = f'(g(x)) * g'(x)`.

A neural network is a composition of many functions. The backward pass computes the gradient of the loss with respect to every weight by applying the chain rule backwards through the composition.

### Key Equations for One Layer

Given `dL/da` (how the loss changes w.r.t. this layer's outputs):

```
dL/dz[i]   = dL/da[i] * activation'(z[i])    # chain rule through activation
dL/dW[i][j] = dL/dz[i] * input[j]             # chain rule through dot product
dL/db[i]    = dL/dz[i]                         # bias gradient
dL/dx[j]    = sum_i( dL/dz[i] * W[i][j] )     # gradient for previous layer
```

### The Softmax + Cross-Entropy Shortcut

When the output uses softmax + cross-entropy, the combined gradient at the output is:

```
dL/dz = softmax(z) - target
```

This is why softmax and cross-entropy are almost always used together. The math works out to something elegant and numerically stable.

### Gradient Checking

You can verify backprop numerically:

```python
numerical_grad = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```

If your analytical gradient matches the numerical one (difference < 1e-5), backprop is correct. This check is in Chapter 5's solution file.

---

## Chapter 6: Training Loop

The training loop is the engine:

```
for each training example:
    1. Forward pass  → get prediction
    2. Compute loss  → measure the mistake
    3. Backward pass → compute gradients
    4. SGD update    → adjust weights
```

### SGD Update

```python
for each weight w:
    w = w - learning_rate * gradient_of_w
```

The learning rate (`lr`, typically 0.001 to 0.1) controls step size. Too large: training diverges. Too small: training takes forever.

"Stochastic" means we update after every single example rather than averaging over the full dataset. It's noisier but often converges faster.

### Convergence

One "epoch" is a full pass through the training data. Watch the average loss per epoch — it should decrease. When it stops decreasing, training has converged.

---

## Chapter 7: Final Project — Letter Classifier

All seven chapters come together. The `LetterClassifier`:

1. Represents each letter as a 26-dimensional one-hot vector
2. Builds a `BackpropNetwork([26, 64, 26])`
3. Trains for 500 epochs with SGD
4. Achieves 95%+ accuracy on all 26 letters

The network has 3,418 parameters — 26×64 weights + 64 biases + 64×26 weights + 26 biases. At 4 bytes each, that's ~13.7 KB. Smaller than most profile pictures.

Run it:

```bash
python3 tutorials/01-mlp-from-scratch/solution/07_final_project.py
```

It takes about 5-10 seconds on a modern laptop.

---

## Common Mistakes

**Forgetting to reset stored activations between forward passes**: if you call `forward()` twice before `backward()`, the second forward overwrites the first's cached values. Backprop then computes gradients for the wrong pass.

**Wrong learning rate**: if loss oscillates wildly or grows, try lr=0.01. If loss barely moves, try lr=0.1.

**Using mutable default arguments**: `def func(v=[])` in Python means the same list object is shared across all calls. Use `None` as the default and create the list inside the function.

**Transposing at the wrong step**: in backprop, the input gradient uses the transposed weight matrix. Missing the transpose gives you wrong shapes and wrong gradients.
