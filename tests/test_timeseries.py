# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import Unittest
import unittest

# Import third party packages
import pandas as pd
import pytz

# Import my package
from scifin.timeseries import timeseries as ts
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#




class TestTimeSeries(unittest.TestCase):
    """
    Tests the class TimeSeries.
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

        
    def test_TimeSeries_init_fail(self):
        
        # Define TimeSeries
        test_df = pd.DataFrame(columns=['g1','g2','g3'], index=['2020-01','2020-02'])
        test_df.loc['2020-01'] = {'g1':1, 'g2':2, 'g3':3}
        test_df.loc['2020-02'] = {'g1':4, 'g2':5, 'g3':6}
        
        # Test Error
        with self.assertRaises(AssertionError):
            ts1 = ts.TimeSeries(df=test_df, tz="Europe/London", unit='£', name="Test a time series")


    def test_TimeSeries_init(self):
        
        # Define TimeSeries
        test_df = pd.DataFrame(columns=['ts'], index=['2020-01','2020-02','2020-03'])
        test_df.loc['2020-01'] = 1.
        test_df.loc['2020-02'] = 2.
        test_df.loc['2020-03'] = 3.
        self.ts1 = ts.TimeSeries(df=test_df, tz="Europe/London", unit='£', name="Test a time series")
        
        # Test attributes values
        self.assertEqual(self.ts1.data.iloc[0,0], 1.)
        self.assertEqual(self.ts1.start_utc, '2020-01')
        self.assertEqual(self.ts1.end_utc, '2020-03')
        self.assertEqual(self.ts1.nvalues, 3)
        self.assertEqual(self.ts1.freq, 'MS')
        self.assertEqual(self.ts1.unit, '£')
        self.assertEqual(self.ts1.tz, "Europe/London")
        self.assertEqual(self.ts1.timezone, pytz.timezone("Europe/London"))
        self.assertEqual(self.ts1.name, "Test a time series")
        self.assertEqual(self.ts1.type, 'TimeSeries')
        
        # Test methods
        self.assertEqual(self.ts1.hist_avg(), 2.0)
        self.assertEqual(self.ts1.hist_std(), 0.816496580927726)
        self.assertEqual(self.ts1.hist_variance(), 0.6666666666666666)
        self.assertEqual(self.ts1.hist_kurtosis(), 1.5)
        self.assertEqual(self.ts1.min(), 1.0)
        self.assertEqual(self.ts1.max(), 3.0)
        self.assertListEqual(self.ts1.percent_change().data.values.flatten().tolist(), [1.0, 0.5])


    def test_CatTimeSeries_init(self):
        
        # Define TimeSeries
        test_df = pd.DataFrame(columns=['ts'], index=['2020-01','2020-02','2020-03'])
        test_df.loc['2020-01'] = 'a'
        test_df.loc['2020-02'] = 'b'
        test_df.loc['2020-03'] = 'c'
        self.cts1 = ts.CatTimeSeries(df=test_df, tz="UTC", unit='$', name="Test a categorical time series")
        
        # Test attributes values
        self.assertEqual(self.cts1.data.iloc[0,0], 'a')
        self.assertEqual(self.cts1.start_utc, '2020-01')
        self.assertEqual(self.cts1.end_utc, '2020-03')
        self.assertEqual(self.cts1.nvalues, 3)
        self.assertEqual(self.cts1.freq, 'MS')
        self.assertEqual(self.cts1.unit, '$')
        self.assertEqual(self.cts1.tz, "UTC")
        self.assertEqual(self.cts1.timezone, pytz.timezone("UTC"))
        self.assertEqual(self.cts1.name, "Test a categorical time series")
        self.assertEqual(self.cts1.type, 'CatTimeSeries')
        

        
    def test_build_from_lists(self):
        
        # Define TimeSeries
        self.ts1 = ts.build_from_lists(list_dates=['2020-01','2020-02','2020-03'], list_values=[1.,2.,3.],
                                       tz="Europe/London", unit='£', name="Test a time series")
        
        # Test attributes values
        self.assertEqual(self.ts1.data.iloc[0,0], 1.)
        self.assertEqual(self.ts1.start_utc, '2020-01')
        self.assertEqual(self.ts1.end_utc, '2020-03')
        self.assertEqual(self.ts1.nvalues, 3)
        self.assertEqual(self.ts1.freq, 'MS')
        self.assertEqual(self.ts1.unit, '£')
        self.assertEqual(self.ts1.tz, "Europe/London")
        self.assertEqual(self.ts1.timezone, pytz.timezone("Europe/London"))
        self.assertEqual(self.ts1.name, "Test a time series")
        self.assertEqual(self.ts1.type, 'TimeSeries')
        
        # Test methods
        self.assertEqual(self.ts1.hist_avg(), 2.0)
        self.assertEqual(self.ts1.hist_std(), 0.816496580927726)
        self.assertEqual(self.ts1.hist_variance(), 0.6666666666666666)
        self.assertEqual(self.ts1.hist_kurtosis(), 1.5)
        self.assertEqual(self.ts1.min(), 1.0)
        self.assertEqual(self.ts1.max(), 3.0)
        self.assertListEqual(self.ts1.percent_change().data.values.flatten().tolist(), [1.0, 0.5])
        
        
        # Define CatTimeSeries
        self.cts1 = ts.build_from_lists(list_dates=['2020-01','2020-02','2020-03'], list_values=['a','b','c'],
                                       tz="UTC", unit='$', name="Test a categorical time series")
        
        # Test attributes values
        self.assertEqual(self.cts1.data.iloc[0,0], 'a')
        self.assertEqual(self.cts1.start_utc, '2020-01')
        self.assertEqual(self.cts1.end_utc, '2020-03')
        self.assertEqual(self.cts1.nvalues, 3)
        self.assertEqual(self.cts1.freq, 'MS')
        self.assertEqual(self.cts1.unit, '$')
        self.assertEqual(self.cts1.tz, "UTC")
        self.assertEqual(self.cts1.timezone, pytz.timezone("UTC"))
        self.assertEqual(self.cts1.name, "Test a categorical time series")
        self.assertEqual(self.cts1.type, 'CatTimeSeries')
        
        
if __name__ == '__main__':
    unittest.main()
    
