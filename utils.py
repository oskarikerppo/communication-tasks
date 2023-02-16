"""Definitions for commonly used utility functions."""

import numpy as np


def matrix_is_rowstochastic(matrix: np.ndarray) -> bool:
    """Check if input matrix is row-stochastic.

    Args:
        matrix (np.ndarray): input matrix

    Returns:
        bool: True if matrix is row-stochastic else False
    """
    for row in matrix:
        if sum(row) != 1:
            return False
    return True


if __name__ == "__main__":
    test_matrix_1 = np.array([[1, 0], [0.5, 0.5]])
    test_matrix_2 = np.array([[1, 0], [0.5, 0.6]])
    print(matrix_is_rowstochastic(test_matrix_1))
    print(matrix_is_rowstochastic(test_matrix_2))
