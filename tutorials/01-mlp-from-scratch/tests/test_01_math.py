"""
Tests for 01_math_foundations.py

Chapter 1 — Math Foundations: vectors, matrices, dot products.
All operations are pure Python, no numpy.

Run with: python3 -m pytest tests/test_level_a/test_01_math.py -v
"""

import importlib
import sys
import os

# 01_math_foundations.py can't be imported with a leading digit, so use importlib
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'solution')
)
math_foundations = importlib.import_module('01_math_foundations')

vector_add           = math_foundations.vector_add
vector_subtract      = math_foundations.vector_subtract
vector_scale         = math_foundations.vector_scale
dot_product          = math_foundations.dot_product
elementwise_multiply = math_foundations.elementwise_multiply
transpose            = math_foundations.transpose
matrix_vector_multiply = math_foundations.matrix_vector_multiply
matrix_multiply      = math_foundations.matrix_multiply


# ---------------------------------------------------------------------------
# vector_add
# ---------------------------------------------------------------------------

def test_vector_add_basic():
    """[1,2,3] + [4,5,6] = [5,7,9]"""
    assert vector_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def test_vector_add_zeros():
    """Adding the zero vector is an identity operation."""
    v = [3, 1, 4, 1, 5]
    assert vector_add(v, [0, 0, 0, 0, 0]) == v


def test_vector_add_negative():
    """Adding a negative vector acts like subtraction."""
    assert vector_add([5, 5, 5], [-5, -5, -5]) == [0, 0, 0]


# ---------------------------------------------------------------------------
# vector_subtract
# ---------------------------------------------------------------------------

def test_vector_subtract_basic():
    """[5,7,9] - [4,5,6] = [1,2,3]"""
    assert vector_subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def test_vector_subtract_self():
    """Any vector minus itself is the zero vector."""
    v = [9, 8, 7]
    assert vector_subtract(v, v) == [0, 0, 0]


# ---------------------------------------------------------------------------
# vector_scale
# ---------------------------------------------------------------------------

def test_vector_scale_basic():
    """2 * [1,2,3] = [2,4,6]"""
    assert vector_scale([1, 2, 3], 2) == [2, 4, 6]


def test_vector_scale_zero():
    """0 * anything = zero vector."""
    assert vector_scale([99, 42, 7], 0) == [0, 0, 0]


def test_vector_scale_one():
    """1 * v = v (multiplicative identity)."""
    v = [3, 1, 4]
    assert vector_scale(v, 1) == v


def test_vector_scale_fractional():
    """Scaling by 0.5 halves each element."""
    assert vector_scale([2, 4, 6], 0.5) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------

def test_dot_product_basic():
    """[1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 32"""
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32


def test_dot_product_orthogonal():
    """Orthogonal unit vectors have a dot product of 0."""
    assert dot_product([1, 0], [0, 1]) == 0


def test_dot_product_self():
    """v · v = sum of squares (squared magnitude)."""
    assert dot_product([3, 4], [3, 4]) == 25  # 9 + 16


def test_dot_product_single_element():
    """Scalar case: [a] · [b] = a*b."""
    assert dot_product([7], [6]) == 42


# ---------------------------------------------------------------------------
# elementwise_multiply (Hadamard product)
# ---------------------------------------------------------------------------

def test_elementwise_multiply_basic():
    """[1,2,3] ⊙ [4,5,6] = [4,10,18]"""
    assert elementwise_multiply([1, 2, 3], [4, 5, 6]) == [4, 10, 18]


def test_elementwise_multiply_zeros():
    """Hadamard product with zeros produces the zero vector."""
    assert elementwise_multiply([5, 6, 7], [0, 0, 0]) == [0, 0, 0]


def test_elementwise_multiply_ones():
    """Hadamard product with ones is identity."""
    v = [3, 1, 4]
    assert elementwise_multiply(v, [1, 1, 1]) == v


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------

def test_transpose_rectangular():
    """[[1,2,3],[4,5,6]] transposed → [[1,4],[2,5],[3,6]]"""
    matrix = [[1, 2, 3], [4, 5, 6]]
    expected = [[1, 4], [2, 5], [3, 6]]
    assert transpose(matrix) == expected


def test_transpose_square():
    """Transposing a 2x2 flips the off-diagonal elements."""
    matrix = [[1, 2], [3, 4]]
    expected = [[1, 3], [2, 4]]
    assert transpose(matrix) == expected


def test_transpose_involution():
    """Transposing twice returns the original matrix."""
    matrix = [[1, 2, 3], [4, 5, 6]]
    assert transpose(transpose(matrix)) == matrix


def test_transpose_single_row():
    """A single-row matrix becomes a column (many rows, 1 col each)."""
    matrix = [[1, 2, 3]]
    expected = [[1], [2], [3]]
    assert transpose(matrix) == expected


# ---------------------------------------------------------------------------
# matrix_vector_multiply
# ---------------------------------------------------------------------------

def test_matrix_vector_multiply_basic():
    """
    [[1,2],[3,4]] @ [5,6]
      row0: 1*5 + 2*6 = 17
      row1: 3*5 + 4*6 = 39
    """
    M = [[1, 2], [3, 4]]
    v = [5, 6]
    assert matrix_vector_multiply(M, v) == [17, 39]


def test_matrix_vector_multiply_identity():
    """Identity matrix times any vector returns the same vector."""
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    v = [7, 8, 9]
    assert matrix_vector_multiply(I, v) == v


def test_matrix_vector_multiply_zeros():
    """Zero matrix times any vector returns the zero vector."""
    Z = [[0, 0], [0, 0]]
    v = [5, 5]
    assert matrix_vector_multiply(Z, v) == [0, 0]


# ---------------------------------------------------------------------------
# matrix_multiply
# ---------------------------------------------------------------------------

def test_matrix_multiply_2x2():
    """
    [[1,2],[3,4]] @ [[5,6],[7,8]]
      [0,0]: 1*5+2*7=19   [0,1]: 1*6+2*8=22
      [1,0]: 3*5+4*7=43   [1,1]: 3*6+4*8=50
    """
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    assert matrix_multiply(A, B) == expected


def test_matrix_multiply_identity():
    """A @ I = A for any square matrix A."""
    A = [[1, 2], [3, 4]]
    I = [[1, 0], [0, 1]]
    assert matrix_multiply(A, I) == A


def test_matrix_multiply_different_shapes():
    """2x3 @ 3x2 → 2x2."""
    A = [[1, 2, 3], [4, 5, 6]]      # 2x3
    B = [[7, 8], [9, 10], [11, 12]] # 3x2
    # row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    # row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    expected = [[58, 64], [139, 154]]
    result = matrix_multiply(A, B)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert result == expected


def test_matrix_multiply_zero_matrix():
    """A @ zero = zero matrix."""
    A = [[1, 2], [3, 4]]
    Z = [[0, 0], [0, 0]]
    expected = [[0, 0], [0, 0]]
    assert matrix_multiply(A, Z) == expected
