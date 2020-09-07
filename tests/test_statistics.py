# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import from standard library
import itertools

# Import Unittest
import unittest

# Import third party packages
import numpy as np
import pandas as pd

# Import my package
from scifin import statistics as st
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#




class TestStatistics(unittest.TestCase):
    """
    Tests the functions in statistics.py.
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        pass
        
    def tearDown(self):
        pass


    def test_covariance_to_correlation(self):
        
        # Test 1 - Arbitrary covariance matrix converted to correlation matrix
        self.corr0 = st.covariance_to_correlation(np.array([[1.5, -0.75], [-0.75, 1.5]]))
        self.corr0_correct = np.array([[ 1. , -0.5], [-0.5,  1. ]])
        
        for i,j in itertools.product(range(self.corr0.shape[0]), range(self.corr0.shape[0])):
            self.assertAlmostEqual(self.corr0[i,j], self.corr0_correct[i,j])
        
        
        # Test 2 - Random covariance matrix converted to covariance matrix
        self.corr1 = st.covariance_to_correlation(st.random_covariance_matrix(3, 5))
        
        for i,j in itertools.product(range(self.corr1.shape[0]), range(self.corr1.shape[0])):
            if i==j:
                self.assertAlmostEqual(self.corr1[i,j], 1.)
            else:
                self.assertEqual(self.corr1[i,j], self.corr1[j,i])
                
        
    def test_eigen_value_vector(self):
        
        # Test 1 - Real valued matrix
        self.cov1 = np.array([[1.5, -0.75], [-0.75, 1.5]])
        self.eval1, self.evec1 = st.eigen_value_vector(self.cov1)
        self.eval1_correct = np.array([[2.25, 0.], [0., 0.75]])
        self.evec1_correct = np.array([[-0.7071067811865475, -0.7071067811865475],
                                       [0.7071067811865475, -0.7071067811865475]])

        for i,j in itertools.product(range(self.cov1.shape[0]), range(self.cov1.shape[0])):
            self.assertEqual(self.eval1[i,j], self.eval1_correct[i,j])
            self.assertEqual(self.evec1[i,j], self.evec1_correct[i,j])

       
       # Test 2 - Complex valued matrix
        self.cov2 = np.array([[1,2+3.j],[2-3.j,4]])
        self.eval2, self.evec2 = st.eigen_value_vector(self.cov2)
        self.eval2_correct = np.array([[6.405124837953327, 0.0], [0.0, -1.4051248379533274]])
        self.evec2_correct = np.array([[-0.7071067811865475, -0.7071067811865475], [0.7071067811865475, -0.7071067811865475]])
            
            
            
        
if __name__ == '__main__':
    unittest.main()
    
