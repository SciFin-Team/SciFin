import pytz

from scifin.timeseries import randomseries as rs


class TestRandomSeries:
    """
    Tests the functions in timeseries.randomseries.py.
    """

    def test_randomseries_const(self):
        # Define a random series
        rs1 = rs.constant(start_date="2020-01-01", end_date="2020-04-01", frequency='M',
                          cst=3, sigma=0., tz="Europe/London", unit='$', name="Cst time series")

        # Test attributes values
        assert rs1.data[0] == 3
        assert str(rs1.start_utc) == '2020-01-31 00:00:00'
        assert str(rs1.end_utc) == '2020-03-31 00:00:00'
        assert rs1.nvalues == 3
        assert rs1.freq == 'M'
        assert rs1.unit == '$'
        assert rs1.tz == "Europe/London"
        assert rs1.timezone == pytz.timezone("Europe/London")
        assert rs1.name == "Cst time series"
