# Created on 2020/7/21

# This module is for importing, transforming and visualizing market data.

# Standard library imports
from datetime import datetime
import itertools
import multiprocessing as mp
from typing import Optional, Union

# Third party imports
from IPython.display import display, clear_output
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
def call_yahoo_finance(arguments: (Union[str, list], str, Union[str, datetime.date], Union[str, datetime.date])
                       ) -> Union[pd.Series, pd.DataFrame, None]:
    """
    Get data from Yahoo, either processing one tick at a time, or a list of them (for multi-processing).

    Parameters
    ----------
    arguments :
      All following arguments packed together:
        - assets : str
          Ticker name we want to extract from Yahoo Finance.
        - feature : str
          The feature name among 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'.
        - start_date : str or datetime
          The start date of extraction.
        - end_date : str or datetime
          The end date of extraction.

    Returns
    -------
    pd.Series or None for single asset call, pd.DataFrame for multi-assets call.
      Data from Yahoo Finance either in the form of a Pandas Series or a Pandas DataFrame.
    """

    # Initializations
    (assets, feature, start_date, end_date) = arguments

    if isinstance(assets, str):
        try:
            resu_call = pdr.get_data_yahoo(assets, start=start_date, end=end_date)[feature]
            return resu_call
        except:
            return None
    else:
        imported_data = pd.DataFrame(data=None, columns=assets)
        for i in range(len(assets)):
            try:
                resu_call = pdr.get_data_yahoo(assets[i], start=start_date, end=end_date)[feature]
                imported_data[assets[i]] = resu_call
            except:
                print(f"Data for {assets[i]} could not be imported.")
        return imported_data


@typechecked
def get_assets_from_yahoo(list_assets: list,
                          feature: str,
                          start_date: Union[str, datetime.date],
                          end_date: Union[str, datetime.date],
                          name: str = "",
                          n_proc: Optional[int] = None,
                          verbose: bool = False
                          ) -> simuldata.Market:
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
    n_proc : int
      Number of processors used in multi-processing.
    verbose : bool
      Verbose option.
    
    Returns
    -------
    Market
      A market made of time series having same index and distinctive names.
    
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
    sorted_assets = list(np.sort(list_assets))
    
    # Initialization
    N = len(sorted_assets)
    counter = 1

    # Make a DataFrame with all assets data (when found)
    # With multi-processing
    if n_proc is not None:
        assets = pd.DataFrame(data=None, columns=None)

        # Prepare parts for multi-processing (if requested)
        parts = np.ceil( np.linspace(0, N, min(n_proc, N)+1) ).astype(int)
        jobs = []
        for i in range(1, len(parts)):
            jobs.append(sorted_assets[parts[i-1]:parts[i]])

        # Do the processing
        pool = mp.Pool(processes = n_proc)
        arguments = [i for i in itertools.zip_longest(jobs, itertools.repeat(feature, len(jobs))
                                                          , itertools.repeat(start_date, len(jobs))
                                                          , itertools.repeat(end_date, len(jobs)))]
        outputs = pool.map(call_yahoo_finance, arguments)

        # Combine results from different processors
        for out_ in outputs:
            assets = pd.concat([assets, out_], axis=1, sort=False)
        pool.close()
        pool.join()

    # Without multi-processing (slow for large amount of assets)
    else:
        assets = pd.DataFrame(data=None, columns=sorted_assets)

        for i in range(N):
            # Print status of execution
            if verbose:
                clear_output(wait=True)
                display("Running... " + str(int(counter/N*100)) + '%')
                counter += 1
            # Get data from Yahoo Finance
            call_resu = call_yahoo_finance((sorted_assets[i], feature, start_date, end_date))
            if call_resu is not None:
                assets[sorted_assets[i]] = call_resu

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


