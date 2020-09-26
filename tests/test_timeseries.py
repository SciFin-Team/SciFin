import pandas as pd
import pytz
import pytest

from scifin.timeseries import timeseries as ts


class TestTimeSeries:
    """
    Tests the class TimeSeries.
    """

    def test_TimeSeries_init_fail(self):
        # Define TimeSeries
        test_df = pd.DataFrame(columns=['g1', 'g2', 'g3'], index=['2020-01', '2020-02'])
        test_df.loc['2020-01'] = {'g1': 1, 'g2': 2, 'g3': 3}
        test_df.loc['2020-02'] = {'g1': 4, 'g2': 5, 'g3': 6}

        # Test Error
        with pytest.raises(AssertionError):
            ts1 = ts.TimeSeries(data=test_df, tz="Europe/London", unit='£', name="Test a time series")

    def test_TimeSeries_init(self):
        # Define TimeSeries
        test_df = pd.DataFrame(columns=['ts'], index=['2020-01', '2020-02', '2020-03'])
        test_df.loc['2020-01'] = 1.
        test_df.loc['2020-02'] = 2.
        test_df.loc['2020-03'] = 3.
        ts1 = ts.TimeSeries(data=test_df, tz="Europe/London", unit='£', name="Test a time series")

        # Test attributes values
        assert ts1.data[0] == 1.
        assert ts1.start_utc == '2020-01'
        assert ts1.end_utc == '2020-03'
        assert ts1.nvalues == 3
        assert ts1.freq == 'MS'
        assert ts1.unit == '£'
        assert ts1.tz == "Europe/London"
        assert ts1.timezone == pytz.timezone("Europe/London")
        assert ts1.name == "Test a time series"
        assert ts1.type == 'TimeSeries'

        # Test methods
        assert ts1.hist_avg() == 2.0
        assert ts1.hist_std() == 0.816496580927726
        assert ts1.hist_variance() == 0.6666666666666666
        assert ts1.hist_kurtosis() == 1.5
        assert ts1.min() == 1.0
        assert ts1.max() == 3.0
        assert ts1.percent_change().data.values.flatten().tolist() == [1.0, 0.5]

    def test_CatTimeSeries_init(self):
        # Define TimeSeries
        test_df = pd.DataFrame(columns=['ts'], index=['2020-01', '2020-02', '2020-03'])
        test_df.loc['2020-01'] = 'a'
        test_df.loc['2020-02'] = 'b'
        test_df.loc['2020-03'] = 'c'
        cts1 = ts.CatTimeSeries(data=test_df, tz="UTC", unit='$', name="Test a categorical time series")

        # Test attributes values
        assert cts1.data[0] == 'a'
        assert cts1.start_utc == '2020-01'
        assert cts1.end_utc == '2020-03'
        assert cts1.nvalues == 3
        assert cts1.freq == 'MS'
        assert cts1.unit == '$'
        assert cts1.tz == "UTC"
        assert cts1.timezone == pytz.timezone("UTC")
        assert cts1.name == "Test a categorical time series"
        assert cts1.type == 'CatTimeSeries'

    def test_build_from_lists(self):
        # Define TimeSeries
        ts1 = ts.build_from_lists(list_dates=['2020-01', '2020-02', '2020-03'], list_values=[1., 2., 3.],
                                  tz="Europe/London", unit='£', name="Test a time series")

        # Test attributes values
        assert ts1.data[0] == 1.
        assert ts1.start_utc == '2020-01'
        assert ts1.end_utc == '2020-03'
        assert ts1.nvalues == 3
        assert ts1.freq == 'MS'
        assert ts1.unit == '£'
        assert ts1.tz == "Europe/London"
        assert ts1.timezone == pytz.timezone("Europe/London")
        assert ts1.name == "Test a time series"
        assert ts1.type == 'TimeSeries'
        # Test methods
        assert ts1.hist_avg() == 2.0
        assert ts1.hist_std() == 0.816496580927726
        assert ts1.hist_variance() == 0.6666666666666666
        assert ts1.hist_kurtosis() == 1.5
        assert ts1.min() == 1.0
        assert ts1.max() == 3.0
        assert ts1.percent_change().data.values.flatten().tolist() == [1.0, 0.5]

        # Define CatTimeSeries
        cts1 = ts.build_from_lists(list_dates=['2020-01', '2020-02', '2020-03'], list_values=['a', 'b', 'c'],
                                   tz="UTC", unit='$', name="Test a categorical time series")

        # Test attributes values
        assert cts1.data[0] == 'a'
        assert cts1.start_utc == '2020-01'
        assert cts1.end_utc == '2020-03'
        assert cts1.nvalues == 3
        assert cts1.freq == 'MS'
        assert cts1.unit == '$'
        assert cts1.tz == "UTC"
        assert cts1.timezone == pytz.timezone("UTC")
        assert cts1.name == "Test a categorical time series"
        assert cts1.type == 'CatTimeSeries'
