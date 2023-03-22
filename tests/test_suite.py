"""Unit tests to check functionality."""

import unittest

import numpy as np

from communication_tasks.communication_matrices import (
    CommunicationMatrix,
    RandomCommunicationMatrix,
)
from utils.utils import matrix_is_rowstochastic, sample_random_row_stochastic_matrix


class TestCommunication(unittest.TestCase):
    """Test suite for communication matrices."""

    invalid_communication_matirx = np.array(
        [[1, 0], [0.5, 0.6]]
    )  # an invalid communication matrix

    k_plus = 0.5 * np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )

    k_matrix = 0.5 * np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )

    k_minus = 0.5 * np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    )

    d_3_1_3 = np.array(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
    )

    a_matrix = 0.5 * np.array([[2, 0, 0], [1, 1, 0], [1, 0, 1]])

    b_matrix = 1 / 3 * np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])

    c_matrix = 0.5 * np.array([[2, 0, 0], [0, 1, 1], [1, 0, 1]])

    d_matrix = 0.5 * np.array([[2, 0, 0], [0, 1, 1], [0, 0, 2]])

    def test_row_stochastic_matrix(self):
        """Test that row-stochastic matrix is row-stochastic."""
        self.assertEqual(matrix_is_rowstochastic(self.k_plus), True, "Should be True")

    def test_random_row_stochastic_matrix(self):
        """Test that random row-stochastic matrix is row-stochastic."""
        self.assertEqual(
            matrix_is_rowstochastic(sample_random_row_stochastic_matrix(shape=(5, 5))),
            True,
            "Should be True",
        )

    def test_not_row_stochastic_matrix(self):
        """Test that non-row-stochastic matrix is not row-stochastic."""
        self.assertEqual(
            matrix_is_rowstochastic(self.invalid_communication_matirx),
            False,
            "Should be False",
        )

    def test_communication_matrix_initialization(self):
        """Test that communication matrix is formed correctly."""
        self.assertEqual(
            matrix_is_rowstochastic(CommunicationMatrix(self.d_3_1_3).matrix),
            True,
            "Should be True",
        )

    def test_random_communication_matrix_initialization(self):
        """Test that communication matrix is formed correctly."""
        self.assertEqual(
            matrix_is_rowstochastic(RandomCommunicationMatrix(shape=(5, 5)).matrix),
            True,
            "Should be True",
        )

    def test_communication_matrix_initialization_fails(self):
        """Test that communication matrix is not initialized for invalid input"""
        with self.assertRaises(Exception):
            CommunicationMatrix(self.invalid_communication_matirx)

    def test_communication_number_of_rows(self):
        """Test that communication matrix has correct number of rows."""
        self.assertEqual(
            CommunicationMatrix(self.k_plus).nrows,
            5,
            "Should be 5",
        )

    def test_communication_number_of_columns(self):
        """Test that communication matrix has correct number of rows."""
        self.assertEqual(
            CommunicationMatrix(self.k_plus).ncols,
            4,
            "Should be 4",
        )

    def test_rank(self):
        """Test that rank is calculated correctly."""
        self.assertEqual(
            CommunicationMatrix(self.k_plus).calculate_rank(),
            4,
            "Should be 4",
        )
        self.assertEqual(
            CommunicationMatrix(self.k_matrix).calculate_rank(),
            3,
            "Should be 3",
        )
        self.assertEqual(
            CommunicationMatrix(self.k_minus).calculate_rank(),
            3,
            "Should be 3",
        )
        self.assertEqual(
            CommunicationMatrix(self.d_3_1_3).calculate_rank(),
            3,
            "Should be 3",
        )
        self.assertEqual(
            CommunicationMatrix(self.a_matrix).calculate_rank(),
            3,
            "Should be 3",
        )
        self.assertEqual(
            CommunicationMatrix(self.b_matrix).calculate_rank(),
            3,
            "Should be 3",
        )
        self.assertEqual(
            CommunicationMatrix(self.c_matrix).calculate_rank(),
            3,
            "Should be 3",
        )
        self.assertEqual(
            CommunicationMatrix(self.d_matrix).calculate_rank(),
            3,
            "Should be 3",
        )

    def test_lambda_max(self):
        """Test that lambda max is calculated correctly."""
        self.assertEqual(
            CommunicationMatrix(self.k_plus).calculate_lambda_max(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.k_matrix).calculate_lambda_max(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.k_minus).calculate_lambda_max(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.d_3_1_3).calculate_lambda_max(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.a_matrix).calculate_lambda_max(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.b_matrix).calculate_lambda_max(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.c_matrix).calculate_lambda_max(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.d_matrix).calculate_lambda_max(),
            5 / 2,
            "Should be 5/2",
        )

    def test_lambda_min(self):
        """Test that lambda min is calculated correctly."""
        self.assertEqual(
            CommunicationMatrix(self.k_plus).calculate_lambda_min(),
            0,
            "Should be 0",
        )
        self.assertEqual(
            CommunicationMatrix(self.k_matrix).calculate_lambda_min(),
            0,
            "Should be 0",
        )
        self.assertEqual(
            CommunicationMatrix(self.k_minus).calculate_lambda_min(),
            0,
            "Should be 0",
        )
        self.assertEqual(
            CommunicationMatrix(self.d_3_1_3).calculate_lambda_min(),
            -1 / 2,
            "Should be -1/2",
        )
        self.assertEqual(
            CommunicationMatrix(self.a_matrix).calculate_lambda_min(),
            -1 / 2,
            "Should be -1/2",
        )
        self.assertEqual(
            CommunicationMatrix(self.b_matrix).calculate_lambda_min(),
            0,
            "Should be 0",
        )
        self.assertEqual(
            CommunicationMatrix(self.c_matrix).calculate_lambda_min(),
            0,
            "Should be 0",
        )
        self.assertEqual(
            CommunicationMatrix(self.d_matrix).calculate_lambda_min(),
            0,
            "Should be 0",
        )

    def test_iota(self):
        """Test that iota is calculated correctly."""
        self.assertEqual(
            CommunicationMatrix(self.k_plus).calculate_iota(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.k_matrix).calculate_iota(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.k_minus).calculate_iota(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.d_3_1_3).calculate_iota(),
            1,
            "Should be 1",
        )
        self.assertEqual(
            CommunicationMatrix(self.a_matrix).calculate_iota(),
            1,
            "Should be 1",
        )
        self.assertEqual(
            CommunicationMatrix(self.b_matrix).calculate_iota(),
            1,
            "Should be 1",
        )
        self.assertEqual(
            CommunicationMatrix(self.c_matrix).calculate_iota(),
            2,
            "Should be 2",
        )
        self.assertEqual(
            CommunicationMatrix(self.d_matrix).calculate_iota(),
            2,
            "Should be 2",
        )
