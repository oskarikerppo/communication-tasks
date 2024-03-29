"""A module for implementing communication tasks."""
from typing import Optional, Tuple

import numpy as np

from communication_tasks import monotones
from utils.utils import matrix_is_rowstochastic, sample_random_row_stochastic_matrix


class CommunicationMatrix:
    """A class which defines all relevant features of communication tasks."""

    def __init__(self, matrix: np.ndarray) -> None:
        if not matrix_is_rowstochastic(matrix):
            raise ValueError("Input matrix is not row-stochastic!")
        self.matrix = matrix
        self.nrows: int = matrix.shape[0]
        self.ncols: int = matrix.shape[1]
        self.rank: Optional[int] = None
        self.lambda_min: Optional[float] = None
        self.lambda_max: Optional[float] = None
        self.iota: Optional[int] = None

    def calculate_rank(self) -> int:
        """Calculate the rank of the communication matrix.

        Returns:
            int: rank of the matrix
        """
        rank_of_matrix = monotones.rank(self.matrix)
        self.rank = rank_of_matrix
        return rank_of_matrix

    def calculate_lambda_min(self) -> float:
        """Calculate the lambda min of the communication matrix.

        Returns:
            float: lambda min of the matrix
        """
        lambda_min = monotones.lambda_min(self.matrix)
        self.lambda_min = lambda_min
        return lambda_min

    def calculate_lambda_max(self) -> float:
        """Calculate the lambda max of the communication matrix.

        Returns:
            float: lambda max of the matrix
        """
        lambda_max = monotones.lambda_max(self.matrix)
        self.lambda_max = lambda_max
        return lambda_max

    def calculate_iota(self) -> int:
        """Calculate the iota of the communication matrix.

        Returns:
            int: maximal number of orthogonal rows of the matrix
        """
        iota = monotones.iota(self.matrix)
        self.iota = iota
        return iota


class RandomCommunicationMatrix(CommunicationMatrix):
    """Initialize a random communication matrix with the given shape."""

    def __init__(self, shape: Tuple[int, int]) -> None:
        matrix = sample_random_row_stochastic_matrix(shape)
        super().__init__(matrix)
