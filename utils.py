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
