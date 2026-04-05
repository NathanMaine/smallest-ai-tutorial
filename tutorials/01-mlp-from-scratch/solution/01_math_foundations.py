"""
Chapter 1 — Math Foundations
=============================

What this module teaches
-------------------------
This is the bedrock of every neural network: vectors and matrices.
Before we can build a neuron, a layer, or a backpropagation loop, we need
to understand the four fundamental operations that power them all:

  1. Vectors — ordered lists of numbers that represent data, weights, biases,
     gradients, and activations.
  2. Matrices — 2-D grids of numbers that represent entire layers of weights.
     A matrix is just a list of vectors.
  3. Dot product — the workhorse of a single neuron. One neuron computes a
     weighted sum of its inputs: output = w · x + b. That's a dot product
     plus a scalar.
  4. Matrix multiplication — the workhorse of a full layer. Passing a batch
     of inputs through a layer is one matrix multiply: Y = W @ X.

Why this matters for neural networks
--------------------------------------
A neural network is, at its core, a chain of matrix multiplications
interleaved with non-linear functions (activations). Understanding these
operations from scratch — without hiding them behind NumPy — gives you the
insight needed to:

  - Debug shape errors (the most common neural network bug)
  - Understand the backpropagation chain rule geometrically
  - Implement custom layers, loss functions, and optimizers
  - Read research papers without getting lost in the notation

Implementation philosophy
--------------------------
Everything here is pure Python — no NumPy, no external dependencies.
Vectors are plain Python lists. Matrices are lists of lists.
Each function uses list comprehensions and built-ins (zip, sum, range).

This deliberate constraint forces clarity: you see exactly what the
operation does, one element at a time. After building these yourself,
you'll have an intuition for what NumPy's vectorised operations are
actually computing under the hood.

Chapter roadmap
---------------
  Chapter 1 (this file):  Math foundations — vectors & matrices
  Chapter 2:              Single neuron — dot product + bias + activation
  Chapter 3:              Forward pass — matrix_vector_multiply through a layer
  Chapter 4:              Activation functions — sigmoid, ReLU, tanh
  Chapter 5:              Loss functions — MSE, cross-entropy
  Chapter 6:              Backpropagation — gradients via the chain rule
"""


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def vector_add(a, b):
    """
    Element-wise vector addition: result[i] = a[i] + b[i].

    In neural networks, this is used constantly:
      - Adding a bias vector to a layer's weighted sum: z = W @ x + b
      - Adding a gradient to accumulated gradients during backprop
      - Combining residual connections (adding skip-connection outputs)

    Parameters
    ----------
    a : list of numbers — first vector
    b : list of numbers — second vector (must be the same length as a)

    Returns
    -------
    list of numbers — element-wise sum

    Example
    -------
    >>> vector_add([1, 2, 3], [4, 5, 6])
    [5, 7, 9]
    """
    return [ai + bi for ai, bi in zip(a, b)]


def vector_subtract(a, b):
    """
    Element-wise vector subtraction: result[i] = a[i] - b[i].

    This is the core of gradient descent. Every weight update looks like:

        w_new = w_old - learning_rate * gradient

    Which is:  new_weights = vector_subtract(old_weights, scaled_gradient)

    Parameters
    ----------
    a : list of numbers — minuend vector
    b : list of numbers — subtrahend vector (same length as a)

    Returns
    -------
    list of numbers — element-wise difference

    Example
    -------
    >>> vector_subtract([5, 7, 9], [4, 5, 6])
    [1, 2, 3]
    """
    return [ai - bi for ai, bi in zip(a, b)]


def vector_scale(v, scalar):
    """
    Multiply every element of a vector by a scalar: result[i] = v[i] * scalar.

    Two critical uses in neural networks:
      1. Applying the learning rate: scaled_grad = vector_scale(gradient, lr)
         Before subtracting a gradient from a weight, we scale it by the
         learning rate (e.g. 0.01) so we take small, controlled steps.
      2. Normalisation: dividing by the vector length to make a unit vector.

    Parameters
    ----------
    v      : list of numbers — the vector to scale
    scalar : number — the scaling factor

    Returns
    -------
    list of numbers — scaled vector

    Example
    -------
    >>> vector_scale([1, 2, 3], 2)
    [2, 4, 6]
    >>> vector_scale([1, 2, 3], 0)
    [0, 0, 0]
    """
    return [vi * scalar for vi in v]


def dot_product(a, b):
    """
    Dot product (inner product): result = sum(a[i] * b[i] for all i).

    This is the fundamental computation of a single artificial neuron.
    A neuron computes:

        output = dot_product(weights, inputs) + bias

    Geometrically, the dot product measures how much two vectors "point
    in the same direction". When a = b, the result is the squared magnitude.
    When the vectors are orthogonal (perpendicular), the result is 0.

    Parameters
    ----------
    a : list of numbers — first vector (e.g. weights)
    b : list of numbers — second vector (e.g. inputs), same length as a

    Returns
    -------
    number — scalar dot product value

    Example
    -------
    >>> dot_product([1, 2, 3], [4, 5, 6])
    32            # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    >>> dot_product([1, 0], [0, 1])
    0             # orthogonal unit vectors
    """
    return sum(ai * bi for ai, bi in zip(a, b))


def elementwise_multiply(a, b):
    """
    Hadamard product (element-wise multiplication): result[i] = a[i] * b[i].

    Also written as a ⊙ b. This is NOT the dot product (which sums the
    products). The Hadamard product returns a vector of the same shape.

    Primary use in neural networks: backpropagation through activations.
    When computing the gradient of the loss with respect to pre-activation
    values, we multiply the upstream gradient by the derivative of the
    activation function — element by element:

        delta = elementwise_multiply(upstream_gradient, activation_derivative)

    Parameters
    ----------
    a : list of numbers — first vector
    b : list of numbers — second vector (same length as a)

    Returns
    -------
    list of numbers — element-wise products

    Example
    -------
    >>> elementwise_multiply([1, 2, 3], [4, 5, 6])
    [4, 10, 18]
    """
    return [ai * bi for ai, bi in zip(a, b)]


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

def transpose(matrix):
    """
    Transpose a matrix: swap rows and columns.

    If matrix has shape (m, n), the transpose has shape (n, m).
    result[j][i] = matrix[i][j].

    Transposing is critical in backpropagation. When the forward pass
    computes:

        output = W @ input          # W is shape (out, in)

    The backward pass (computing gradient w.r.t. input) requires:

        grad_input = transpose(W) @ grad_output   # W^T is shape (in, out)

    Parameters
    ----------
    matrix : list of lists — a 2-D matrix with m rows and n columns

    Returns
    -------
    list of lists — transposed matrix with n rows and m columns

    Example
    -------
    >>> transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    """
    if not matrix or not matrix[0]:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


def matrix_vector_multiply(matrix, vector):
    """
    Multiply a matrix by a column vector: result = matrix @ vector.

    This is the core operation of a single neural network layer's forward
    pass. A layer with weight matrix W (shape: out_features × in_features)
    transforms an input vector x (length: in_features) into an output
    vector (length: out_features):

        output = matrix_vector_multiply(W, x)   # then add bias + activation

    Each element of the output is the dot product of one row of the matrix
    with the input vector:

        result[i] = dot_product(matrix[i], vector)

    Parameters
    ----------
    matrix : list of lists — shape (m, n)
    vector : list of numbers — length n

    Returns
    -------
    list of numbers — length m (one value per row of the matrix)

    Example
    -------
    >>> matrix_vector_multiply([[1, 2], [3, 4]], [5, 6])
    [17, 39]    # row0: 1*5+2*6=17,  row1: 3*5+4*6=39
    """
    return [dot_product(row, vector) for row in matrix]


def matrix_multiply(a, b):
    """
    General matrix multiplication: result = a @ b.

    If a has shape (m, k) and b has shape (k, n), the result has shape (m, n).

        result[i][j] = dot_product(row i of a, column j of b)

    This is how we process batches in neural networks — rather than calling
    matrix_vector_multiply once per sample, we stack inputs into a matrix
    and do one matrix multiply for the entire batch. It's also used in:

      - Multi-head attention (Q @ K^T, then softmax @ V)
      - Computing weight gradients in backprop: dW = grad_output @ input^T
      - Any operation that transforms a set of feature vectors simultaneously

    Parameters
    ----------
    a : list of lists — shape (m, k)
    b : list of lists — shape (k, n)

    Returns
    -------
    list of lists — shape (m, n)

    Example
    -------
    >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[19, 22], [43, 50]]
    """
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    # Transpose b so we can use dot_product on rows
    b_t = transpose(b)
    return [[dot_product(a[i], b_t[j]) for j in range(n)] for i in range(m)]


# ---------------------------------------------------------------------------
# Demo — run with:  python3 phase1-from-scratch/level-a-abcs/01_math_foundations.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 1 — Math Foundations Demo")
    print("Pure Python, no libraries")
    print("=" * 60)

    # --- Vectors ---
    print("\n--- Vector Operations ---")

    a = [1, 2, 3]
    b = [4, 5, 6]

    result = vector_add(a, b)
    print(f"vector_add({a}, {b}) = {result}")
    # Used for: adding bias vectors, accumulating gradients, residual connections

    result = vector_subtract(b, a)
    print(f"vector_subtract({b}, {a}) = {result}")
    # Used for: gradient descent weight updates

    result = vector_scale(a, 2)
    print(f"vector_scale({a}, 2) = {result}")
    # Used for: applying learning rate, normalisation

    result = dot_product(a, b)
    print(f"dot_product({a}, {b}) = {result}")
    # Core neuron computation: w · x (weighted sum of inputs)

    result = elementwise_multiply(a, b)
    print(f"elementwise_multiply({a}, {b}) = {result}")
    # Hadamard product — used in backprop through activations

    # --- Matrices ---
    print("\n--- Matrix Operations ---")

    W = [[1, 2, 3],
         [4, 5, 6]]  # 2 x 3 weight matrix (2 neurons, 3 inputs each)
    x = [7, 8, 9]    # input vector (3 features)

    Wt = transpose(W)
    print(f"transpose([[1,2,3],[4,5,6]]) =")
    for row in Wt:
        print(f"  {row}")
    # Shape changes: (2,3) → (3,2) — used in backprop to compute grad_input

    result = matrix_vector_multiply(W, x)
    print(f"\nmatrix_vector_multiply(W, {x}) = {result}")
    # W @ x: the forward pass through one layer (before bias + activation)
    # row0: 1*7+2*8+3*9 = 50,  row1: 4*7+5*8+6*9 = 122

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = matrix_multiply(A, B)
    print(f"\nmatrix_multiply([[1,2],[3,4]], [[5,6],[7,8]]) =")
    for row in C:
        print(f"  {row}")
    # General matmul — batch forward passes, weight gradient computation

    # --- A tiny neuron by hand ---
    print("\n--- A Single Neuron From Scratch ---")
    weights = [0.5, -0.3, 0.8]
    inputs  = [1.0,  2.0, 3.0]
    bias    = 0.1
    pre_activation = dot_product(weights, inputs) + bias
    print(f"weights:        {weights}")
    print(f"inputs:         {inputs}")
    print(f"bias:           {bias}")
    print(f"dot(w, x) + b = {pre_activation:.4f}")
    print("(Pass this to an activation function → neuron output)")

    print("\n" + "=" * 60)
    print("All operations demonstrated. Chapter 2: Single Neuron.")
    print("=" * 60)
