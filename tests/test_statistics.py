import itertools
import pytest

import numpy as np

from scifin import statistics as st


class TestStatistics:
    """
    Tests the functions in statistics.py.
    """

    def test_covariance_to_correlation(self):
        # Test 1 - Arbitrary covariance matrix converted to correlation matrix
        corr0 = st.covariance_to_correlation(np.array([[1.5, -0.75], [-0.75, 1.5]]))
        corr0_correct = np.array([[1., -0.5], [-0.5,  1.]])
        
        for i, j in itertools.product(range(corr0.shape[0]), range(corr0.shape[0])):
            assert corr0[i, j] == pytest.approx(corr0_correct[i, j])
        
        # Test 2 - Random covariance matrix converted to covariance matrix
        corr1 = st.covariance_to_correlation(st.random_covariance_matrix(3, 5))
        
        for i, j in itertools.product(range(corr1.shape[0]), range(corr1.shape[0])):
            if i == j:
                assert corr1[i, j] == pytest.approx(1.)
            else:
                assert corr1[i, j] == corr1[j, i]

    def test_eigen_value_vector(self):
        # Test 1 - Real valued matrix
        cov1 = np.array([[1.5, -0.75], [-0.75, 1.5]])
        eval1, evec1 = st.eigen_value_vector(cov1)
        eval1_correct = np.array([[2.25, 0.], [0., 0.75]])
        evec1_correct = np.array([[-0.7071067811865475, -0.7071067811865475],
                                       [0.7071067811865475, -0.7071067811865475]])

        for i, j in itertools.product(range(cov1.shape[0]), range(cov1.shape[0])):
            assert eval1[i, j] == eval1_correct[i, j]
            assert evec1[i, j] == evec1_correct[i, j]

        # Test 2 - Complex valued matrix
        cov2 = np.array([[1,2+3.j],[2-3.j,4]])
        eval2, evec2 = st.eigen_value_vector(cov2)
        eval2_correct = np.array([[6.405124837953327, 0.0], [0.0, -1.4051248379533274]])
        evec2_correct = np.array([[-0.7071067811865475, -0.7071067811865475], [0.7071067811865475, -0.7071067811865475]])
