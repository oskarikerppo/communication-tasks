"""Module implementing known ultraweak monotone functions."""

import numpy as np


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
