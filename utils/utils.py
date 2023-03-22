"""Definitions for commonly used utility functions."""

from math import isclose
from typing import Tuple

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


def exp_distribution(number: float) -> float:
    """Sample from exponential distribution.

    Args:
        number (float): random uniform number from [0, 1)

    Returns:
        float: -1 times natural logarithm of number
    """
    return -1 * np.log(number)


def sample_random_row_stochastic_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Sample a random row-stochastic matrix. First generate a matrix with each entry
    randomly sampled from the interval [0, 1). Randomly sampling from probability
    distribution is the same as sampling from an n-simplex. Thus an exponential
    distribution must be used. Finally normalize each row to 1.

    Args:
        shape (Tuple[int, int]): The dimensions of the desired matrix.

    Returns:
        np.ndarray: Randomly sampled row-stochastic matrix.
    """
    random_matrix = np.random.rand(*shape)
    random_matrix = exp_distribution(random_matrix)
    random_matrix = (random_matrix.T / random_matrix.sum(axis=1)).T
    return random_matrix
