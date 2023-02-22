"""Unit tests to check functionality."""

import unittest

import numpy as np

from communication_matrices import CommunicationMatrix
from utils import matrix_is_rowstochastic


class TestCommunication(unittest.TestCase):
    """Test suite for communication matrices."""

    test_matrix_1 = np.array([[1, 0], [0.5, 0.5]])  # a valid communication matrix
    test_matrix_2 = np.array([[1, 0], [0.5, 0.6]])  # an invalid communication matrix
    test_matrix_3 = np.array(
        [
            [0.5, 0.5, 0, 0],
            [0.5, 0, 0.5, 0],
            [0.5, 0, 0, 0.5],
            [0, 0.5, 0.5, 0],
            [0, 0.5, 0, 0.5],
            [0, 0, 0.5, 0.5],
        ]
    )

    def test_row_stochastic_matrix(self):
        """Test that row-stochastic matrix is row-stochastic."""
        self.assertEqual(
            matrix_is_rowstochastic(self.test_matrix_1), True, "Should be True"
        )

    def test_not_row_stochastic_matrix(self):
        """Test that non-row-stochastic matrix is not row-stochastic."""
        self.assertEqual(
            matrix_is_rowstochastic(self.test_matrix_2), False, "Should be False"
        )

    def test_communication_matrix_initialization(self):
        """Test that communication matrix is formed correctly."""
        self.assertEqual(
            matrix_is_rowstochastic(CommunicationMatrix(self.test_matrix_1).matrix),
            True,
            "Should be True",
        )

    def test_communication_matrix_initialization_fails(self):
        """Test that communication matrix is not initialized for invalid input"""
        with self.assertRaises(Exception):
            CommunicationMatrix(self.test_matrix_2)

    def test_rank(self):
        """Test that rank is calculated correctly."""
        self.assertEqual(
            CommunicationMatrix(self.test_matrix_3).calculate_rank(),
            4,
            "Should be 4",
        )

    def test_lambda_max(self):
        """Test that lambda max is calculated correctly."""
        self.assertEqual(
            CommunicationMatrix(self.test_matrix_1).calculate_lambda_max(),
            1.5,
            "Should be 1.5",
        )

    def test_lambda_min(self):
        """Test that lambda min is calculated correctly."""
        self.assertEqual(
            CommunicationMatrix(self.test_matrix_1).calculate_lambda_min(),
            -0.5,
            "Should be -0.5",
        )


if __name__ == "__main__":
    unittest.main()
