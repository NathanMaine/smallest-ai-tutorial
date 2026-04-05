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
operation does, one element at a time.
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
    raise NotImplementedError("Your turn! Hint: use zip(a, b) and a list comprehension.")


def vector_subtract(a, b):
    """
    Element-wise vector subtraction: result[i] = a[i] - b[i].

    This is the core of gradient descent. Every weight update looks like:
        w_new = w_old - learning_rate * gradient

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
    raise NotImplementedError("Your turn!")


def vector_scale(v, scalar):
    """
    Multiply every element of a vector by a scalar: result[i] = v[i] * scalar.

    Critical uses:
      1. Applying the learning rate: scaled_grad = vector_scale(gradient, lr)
      2. Normalisation

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
    """
    raise NotImplementedError("Your turn!")


def dot_product(a, b):
    """
    Dot product (inner product): result = sum(a[i] * b[i] for all i).

    This is the fundamental computation of a single artificial neuron:
        output = dot_product(weights, inputs) + bias

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
    """
    raise NotImplementedError("Your turn! Hint: multiply corresponding elements, then sum.")


def elementwise_multiply(a, b):
    """
    Hadamard product (element-wise multiplication): result[i] = a[i] * b[i].

    This is NOT the dot product (which sums the products). The Hadamard
    product returns a vector of the same shape.

    Primary use: backpropagation through activation functions.

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
    raise NotImplementedError("Your turn!")


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

def transpose(matrix):
    """
    Transpose a matrix: swap rows and columns.

    If matrix has shape (m, n), the transpose has shape (n, m).
    result[j][i] = matrix[i][j].

    Critical in backpropagation: when forward pass computes output = W @ input,
    the backward pass (gradient w.r.t. input) requires transpose(W) @ grad.

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
    raise NotImplementedError("Your turn! Hint: use len(matrix), len(matrix[0]), and nested loops.")


def matrix_vector_multiply(matrix, vector):
    """
    Multiply a matrix by a column vector: result = matrix @ vector.

    This is the core operation of a single neural network layer's forward pass:
        output = matrix_vector_multiply(W, x)

    Each element of the output is the dot product of one row of the matrix
    with the input vector.

    Parameters
    ----------
    matrix : list of lists — shape (m, n)
    vector : list of numbers — length n

    Returns
    -------
    list of numbers — length m

    Example
    -------
    >>> matrix_vector_multiply([[1, 2], [3, 4]], [5, 6])
    [17, 39]    # row0: 1*5+2*6=17,  row1: 3*5+4*6=39
    """
    raise NotImplementedError("Your turn! Hint: use dot_product on each row.")


def matrix_multiply(a, b):
    """
    General matrix multiplication: result = a @ b.

    If a has shape (m, k) and b has shape (k, n), the result has shape (m, n).
        result[i][j] = dot_product(row i of a, column j of b)

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
    raise NotImplementedError("Your turn! Hint: transpose b first, then use dot_product on rows.")


# ---------------------------------------------------------------------------
# Demo — run with:  python3 starter_code/01_math_foundations.py
# (after implementing the functions above)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing your implementations...")
    print("Implement each function above, then run this file to verify.")

    # --- Test 1: vector_add ---
    result = vector_add([1, 2, 3], [4, 5, 6])
    assert result == [5, 7, 9], f"vector_add failed: got {result}"
    print("vector_add: PASS")

    # --- Test 2: dot_product ---
    result = dot_product([1, 2, 3], [4, 5, 6])
    assert result == 32, f"dot_product failed: got {result}"
    print("dot_product: PASS")

    # --- Test 3: matrix_vector_multiply ---
    M = [[1, 2], [3, 4]]
    result = matrix_vector_multiply(M, [5, 6])
    assert result == [17, 39], f"matrix_vector_multiply failed: got {result}"
    print("matrix_vector_multiply: PASS")

    print("\nAll tests passed! Run the solution to see a full demo.")
