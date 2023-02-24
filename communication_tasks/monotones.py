"""Module implementing known ultraweak monotone functions."""

import itertools

import numpy as np

from utils.utils import vector_set_is_orthogonal


def rank(matrix: np.ndarray) -> int:
    """Return the rank of a matrix.

    Args:
        matrix (np.ndarray): 2-dimensional matrix

    Returns:
        int: the rank of the matrix
    """
    return np.linalg.matrix_rank(matrix)


def lambda_min(matrix: np.ndarray) -> float:
    """Calulate the lambda min monotone for a matrix.

    Args:
        matrix (np.ndarray): input matrix

    Returns:
        float: lambda min of the matrix
    """
    value = 0.0
    matrix_transpose = matrix.T
    for row in matrix_transpose:
        value -= min(row)
    return value


def lambda_max(matrix: np.ndarray) -> float:
    """Calulate the lambda max monotone for a matrix.

    Args:
        matrix (np.ndarray): input matrix

    Returns:
        float: lambda max of the matrix
    """
    value = 0.0
    matrix_transpose = matrix.T
    for row in matrix_transpose:
        value += max(row)
    return value


def iota(matrix: np.ndarray) -> int:
    """Calculate the iota monotone for a matrix.
    Iota is defined as the maximal number of orthogonal rows.

    Args:
        matrix (np.ndarray): the input matrix

    Returns:
        int: maximal number of orthogonal rows
    """
    idx_list = list(range(len(matrix)))
    for k in range(len(matrix), 1, -1):
        combinations = itertools.combinations(idx_list, k)
        for combination in combinations:
            if vector_set_is_orthogonal(matrix[(list(combination))]):
                return k
    return 1
