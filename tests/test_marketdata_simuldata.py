# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import Unittest
import unittest

# Import third party packages
import pandas as pd

# Import my package
from scifin.marketdata import simuldata as sd
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#




class TestMarket(unittest.TestCase):
    """
    Tests the class Market.
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

        
    def test_Market_init(self):
        
        # Define Market
        test_df = pd.DataFrame(columns=['g1','g2','g3'], index=['date1','date2'])
        test_df.loc['date1'] = {'g1':1, 'g2':2, 'g3':3}
        test_df.loc['date2'] = {'g1':4, 'g2':5, 'g3':6}
        self.m1 = sd.Market(df=test_df, tz="Asia/Tokyo", units=['a','b','c'], name="MyMarket")
        
        # Test attributes values
        self.assertEqual(self.m1.data.shape, (2,3))
        self.assertEqual(self.m1.start_utc, 'date1')
        self.assertEqual(self.m1.end_utc, 'date2')
        self.assertEqual(self.m1.dims, (2,3))
        self.assertEqual(self.m1.tz, "Asia/Tokyo")
        self.assertListEqual(self.m1.units, ['a','b','c'])
        self.assertEqual(self.m1.freq, 'Unknown')
        self.assertEqual(self.m1.name, "MyMarket")
        
        # Test methods
        self.assertTrue(self.m1.is_index_valid())
        
    

class TestSimuldata(unittest.TestCase):
    """
    Tests the non-class functions in Simuldata.py.
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

        
    def test_create_market_returns(self):
        # Create market
        self.m1 = sd.create_market_returns(r_ini=100., drift=0.1, sigma=0.2, n_years=2, steps_per_year=12, n_components=3,
                                           date="2019-12-31", date_type="end", interval_type="M", tz='Europe/Paris',
                                           units=['a','b','c'], name="Random Market Returns")
        # Test attributes values
        self.assertEqual(self.m1.data.shape, (25,3))
        self.assertEqual(self.m1.start_utc, pd.Period('2017-12', 'M'))
        self.assertEqual(self.m1.end_utc, pd.Period('2019-11', 'M'))
        self.assertEqual(self.m1.dims, (25,3))
        self.assertEqual(self.m1.tz, 'Europe/Paris')
        self.assertListEqual(self.m1.units, ['a','b','c'])
        self.assertEqual(self.m1.freq, 'Unknown')
        self.assertEqual(self.m1.name, "Random Market Returns")
        
        
    
    
if __name__ == '__main__':
    unittest.main()
    
