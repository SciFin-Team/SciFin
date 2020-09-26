import pandas as pd

from scifin.marketdata import simuldata as sd


class TestMarket:
    """
    Tests the class Market.
    """

    def test_Market_init(self):
        # Define Market
        test_df = pd.DataFrame(columns=['g1', 'g2', 'g3'], index=['date1', 'date2'])
        test_df.loc['date1'] = {'g1': 1, 'g2': 2, 'g3': 3}
        test_df.loc['date2'] = {'g1': 4, 'g2': 5, 'g3': 6}
        m1 = sd.Market(df=test_df, tz="Asia/Tokyo", units=['a', 'b', 'c'], name="MyMarket")

        # Test attributes values
        assert m1.data.shape == (2, 3)
        assert m1.start_utc == 'date1'
        assert m1.end_utc == 'date2'
        assert m1.dims == (2, 3)
        assert m1.tz == "Asia/Tokyo"
        assert m1.units == ['a', 'b', 'c']
        assert m1.freq == 'Unknown'
        assert m1.name == "MyMarket"

        # Test methods
        assert m1.is_index_valid()


class TestSimuldata:
    """
    Tests the non-class functions in Simuldata.py.
    """

    def test_create_market_returns(self):
        # Create market
        m1 = sd.create_market_returns(r_ini=100., drift=0.1, sigma=0.2, n_years=2, steps_per_year=12,
                                      n_components=3,
                                      date="2019-12-31", date_type="end", interval_type="M", tz='Europe/Paris',
                                      units=['a', 'b', 'c'], name="Random Market Returns")
        # Test attributes values
        assert m1.data.shape == (25, 3)
        assert m1.start_utc == pd.Period('2017-12', 'M')
        assert m1.end_utc == pd.Period('2019-11', 'M')
        assert m1.dims == (25, 3)
        assert m1.tz == 'Europe/Paris'
        assert m1.units == ['a', 'b', 'c']
        assert m1.freq == 'Unknown'
        assert m1.name == "Random Market Returns"
