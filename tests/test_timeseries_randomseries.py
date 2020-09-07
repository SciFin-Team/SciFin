# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import Unittest
import unittest

# Standard library imports
from datetime import datetime

# Import third party packages
import pandas as pd
import pytz

# Import my package
from scifin.timeseries import randomseries as rs
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#




class TestRandomSeries(unittest.TestCase):
    """
    Tests the functions in timeseries.randomseries.py.
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

        
    def test_randomseries_const(self):
        
        # Define a random series
        self.rs1 = rs.constant(start_date="2020-01-01", end_date="2020-04-01", frequency='M',
                      cst=3, sigma=0., tz="Europe/London", unit='$', name="Cst time series")
        
        # Test attributes values
        self.assertEqual(self.rs1.data[0], 3)
        self.assertEqual(str(self.rs1.start_utc), '2020-01-31 00:00:00')
        self.assertEqual(str(self.rs1.end_utc), '2020-03-31 00:00:00')
        self.assertEqual(self.rs1.nvalues, 3)
        self.assertEqual(self.rs1.freq, 'M')
        self.assertEqual(self.rs1.unit, '$')
        self.assertEqual(self.rs1.tz, "Europe/London")
        self.assertEqual(self.rs1.timezone, pytz.timezone("Europe/London"))
        self.assertEqual(self.rs1.name, "Cst time series")

    
if __name__ == '__main__':
    unittest.main()
    
