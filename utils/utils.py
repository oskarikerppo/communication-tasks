"""Definitions for commonly used utility functions."""

from math import isclose

import numpy as np


def matrix_is_rowstochastic(matrix: np.ndarray) -> bool:
    """Check if input matrix is row-stochastic.

    Args:
        matrix (np.ndarray): input matrix

    Returns:
        bool: True if matrix is row-stochastic else False
    """
    for row in matrix:
        if not isclose(sum(row), 1):
            return False
    return True


def vector_set_is_orthogonal(matrix: np.ndarray) -> bool:
    """Return true if the input matrix is orthogonal, i.e., all rows
    of the input matrix are orthogonal.

    Args:
        matrix (np.ndarray): the input matrix

    Returns:
        bool: True if matrix is orthogonal, False otherwise
    """
    matrix_product = matrix @ matrix.T
    matrix_product[matrix_product.nonzero()] = 1  # normalize nonzero entries
    id_matrix = np.identity(len(matrix_product))
    return np.allclose(matrix_product, id_matrix)
