# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import Unittest
import unittest

# Import third party packages
import pandas as pd

# Import my package
from scifin.marketdata import marketdata as md
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#




class TestMarketData(unittest.TestCase):
    """
    Tests the class Individual.
    """
    
    @classmethod
    def setUpClass(cls):
        #print('setupClass')
        pass
    
    @classmethod
    def tearDownClass(cls):
        #print('tearDownClass')
        pass
    
    def setUp(self):
        #print('setUp')
        pass
        
    def tearDown(self):
        # print('tearDown')
        pass

    

    
if __name__ == '__main__':
    unittest.main()
    
