# Created on 2020/7/21

# This module is for importing, transforming and visualizing market data.

# Standard library imports
from datetime import datetime
from typing import Union

# Third party imports
# from IPython.display import display, clear_output
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from typeguard import typechecked

# Local application imports
from .. import exceptions
from . import simuldata


#---------#---------#---------#---------#---------#---------#---------#---------#---------#

@typechecked
def get_sp500_tickers() -> list:
    """
    Gets the SP500 tickers from Wikipedia.
    
    Returns
    -------
    List of str
      The list of tickers from S&P 500.
    """
    
    # Getting the raw data
    try:
        tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                               header=0)[0]['Symbol'].tolist()
    except :
        raise exceptions.AccessError("Could not access the page or extract data.")

    return tickers


@typechecked
def get_assets_from_yahoo(list_assets: list,
                          feature: str,
                          start_date: Union[str, datetime.date],
                          end_date: Union[str, datetime.date],
                          name: str=""):
    """
    Extracts values associated to a feature for a list of assets
    between 2 dates, using Yahoo Finance data.
    
    Parameters
    ----------
    list_assets : list of str
      The list of ticker names we want to extract from Yahoo Finance.
    feature : str
      The feature name among 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'.
    start_date : str or datetime
      The start date of extraction.
    end_date : str or datetime
      The end date of extraction.
    
    Returns
    -------
    Market
      A market made of timeseries having same index and distinctive names.
    
    Raises
    ------
    AssertionError
      When the chosen feature is not in the allowed list.
    
    Notes
    -----
      The choices for the feature provided by Yahoo Finance are:
      'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'.
    
      Learn more about pandas_datareader on:
      https://pydata.github.io/pandas-datareader/stable/index.html
    
    Examples
    --------
      None
    """
    
    # Check the feature is right
    try:
        assert(feature in ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
    except AssertionError:
        raise AssertionError("Feature must be one of the following: \
        'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'.")
    
    # Sort list
    listassets = np.sort(list_assets)
    
    # Initialization
    assets = pd.DataFrame(data=None, columns=listassets)
    N = len(listassets)
    counter = 1
    
    # Make DataFrame
    for i in range(N):
        # print(i)
        
        # Printing status of execution
        # clear_output(wait=True)
        # display("Running... " + str(int(counter/N*100)) + '%')
        print("Running... " + str(int(counter/N*100)) + '%')
        counter += 1
        
        try:
            tmp = pdr.get_data_yahoo(listassets[i], start=start_date, end=end_date)[feature]
            assets[listassets[i]] = tmp
        except:
            print(listassets[i], " could not be imported.")

    # Make Market
    market = simuldata.Market(df=assets, name=name)
    
    return market


@typechecked
def market_EWindex(market: simuldata.Market, name: str="Market EW Index") -> pd.DataFrame:
    """
    Sums all assets to make an index, corresponds to the Equally-Weighed (EW) index.
    
    The formula for the weights here is:
    w_i = c / N for all i
    and we choose c = N so that \sum_i w_i = N.
    
    Thus we get the value at time t of the whole index:
    M_t = \sum_i w_i m_{ti} = \sum_i m_{ti}
    
    Parameters
    ----------
    market : Market
      The market we use to sum values.
    name: str
      Other name for the index.

    Returns
    -------
    DataFrame
      A pandas data frame with the EW index values.
    """
    
    df = pd.DataFrame(market.data.sum(axis=1))
    df.columns = [name]
    
    return df


@typechecked
def get_marketcap_today(market: simuldata.Market) -> pd.Series:
    """
    Returns the market capitalization as it is today.
    
    Parameters
    ----------
    market : Market
      The market we extract tickers from.
    
    Returns
    -------
    Pandas Series
      A pandas series with tickers as index and market cap values.
    """
    
    # Extracting the quotes from the market columns names
    marketcap_today = pdr.data.get_quote_yahoo(market.data.columns)['marketCap']
    
    # Creating a Pandas Series from them
    marketcap = pd.Series(index=marketcap_today.index, data=marketcap_today)
    
    return marketcap


@typechecked
def market_CWindex(market: simuldata.Market, marketcap: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Function that returns the Cap-Weighted (CW) index associated
    with the assets of a market.
    
    We compute the total return at time t, called R_t as:
    R_t = \sum_{i=1}^N v_i^t r_i^t / (\sum_{j=1}^N v_j^t)
    
    And since we only have data of the v_j's today we compute it is:
    R_t = \sum_{i=1}^N v_i^today r_i^t / (\sum_{j=1}^N v_j^today)
    
    Parameters
    ----------
    market : Market
      The market we use to sum values.
    marketcap : DataFrame
      The market capitalization we use to compute weights.
    
    Returns
    -------
    DataFrame
      A pandas data frame with the CW index values.
    """
    
    # Initialization
    if (set(market.data.columns) != set(marketcap.index)):
        print(market.data.columns)
        print(marketcap.index)
        raise IndexError("Error: the two data sources need to have same columns.")
    Nassets = market.dims[1]
    
    # Computing weighted returns
    M = Nassets * (market.data * marketcap / marketcap.sum()).sum(axis=1)
    market_index_df = pd.DataFrame(data=M,
                                   index=market.data.index,
                                   columns=["Market CW Index (Cap from last day)"])
    market_index_df.columns = ['CW sum of assets']
    
    return market_index_df



#---------#---------#---------#---------#---------#---------#---------#---------#---------#


