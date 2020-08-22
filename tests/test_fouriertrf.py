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
from scifin import fouriertrf as ft
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#




class TestFouriertrf(unittest.TestCase):
    """
    Tests the functions in fouriertrf.py.
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


    
    
    
        
if __name__ == '__main__':
    unittest.main()
    
