# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import Unittest
import unittest

# Import third party packages
import numpy as np
import pandas as pd

# Import my package
from scifin import timeseries as ts
from scifin.classifier import classifier as cl
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#




class TestDistances(unittest.TestCase):
    """
    Tests the functions euclidean_distance() and dtw_distance().
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

        
    def test_distances_value(self):
        
        # Build two constant time series
        rs1 = ts.constant(start_date="2000-01-01", end_date="2020-01-01", frequency='Y', cst=1.)
        rs2 = ts.constant(start_date="2000-01-01", end_date="2020-01-01", frequency='Y', cst=2.)
        
        # Test distances are the same
        self.assertEqual(cl.euclidean_distance(rs1,rs2), np.sqrt(rs1.nvalues))
        self.assertEqual(cl.dtw_distance(rs1,rs2), np.sqrt(rs1.nvalues))

        
    def test_order_irrelevance_contant_ts(self):
        
        # Build two constant time series
        rs1 = ts.constant(start_date="2000-01-01", end_date="2020-01-01", frequency='Y', cst=1.)
        rs2 = ts.constant(start_date="2000-01-01", end_date="2020-01-01", frequency='Y', cst=2.)
        
        # Test that order of time series is irrelevant
        self.assertEqual(cl.euclidean_distance(rs1,rs2), cl.euclidean_distance(rs1,rs2))
        self.assertEqual(cl.dtw_distance(rs1,rs2), cl.dtw_distance(rs1,rs2))
        
        
    def test_order_irrelevance_random_ts(self):
        
        # Build two random series
        rs1 = ts.auto_regressive(start_date="2000-01-01", end_date="2020-01-01", frequency='M',
                                 start_values=[1.], cst=1., order=1, coeffs=[0.3], sigma=0.1)

        rs2 = ts.auto_regressive(start_date="2000-01-01", end_date="2020-01-01", frequency='M',
                                 start_values=[1.], cst=1., order=1, coeffs=[0.35], sigma=0.1)

        # Test that order of time series is irrelevant
        self.assertEqual(cl.euclidean_distance(rs1,rs2), cl.euclidean_distance(rs1,rs2))
        self.assertEqual(cl.dtw_distance(rs1,rs2), cl.dtw_distance(rs1,rs2))
        
        
        
if __name__ == '__main__':
    unittest.main()
    

