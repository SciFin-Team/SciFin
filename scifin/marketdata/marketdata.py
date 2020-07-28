# Created on 2020/7/21

# This module is for importing, transforming and visualizing market data.

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import random as random
import matplotlib.pyplot as plt

import bs4 as bs
import pickle
import requests
import pandas_datareader as pdr
from IPython.display import display, clear_output

import timeseries.timeseries as ts


def scrape_sp500_tickers():
    """
    Function that scrapes the SP500 from Wikipedia, using Beautiful Soup package.
    """
    
    # Getting the raw data
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    
    # Saving tables
    tickers = []
    
    # Looping through
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker[:-1])
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers



def get_assets_from_yahoo(list_assets, feature, start_date, end_date):
    """
    Function which extracts values associated to a feature for a list of assets between 2 dates, using Yahoo Finance data.
    
    The choices for the feature provided by Yahoo Finance are: 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'.
    
    Returns a list of timeseries all having the same index and distinctive names.
    """
    
    # Check the feature is right
    available_features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
    try:
        assert(feature in available_features)
    except AssertionError:
        raise(AssertionError("Feature must be one of the following: 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'."))
    
    # Sort list
    listassets = np.sort(list_assets)
    
    # Initialization
    assets = pd.DataFrame(data=None, columns=listassets)
    N = len(listassets)
    counter = 1
    
    # loop
    for i in range(N):
        print(i)
        # Printing status of execution
        clear_output(wait=True)
        display("Running... " + str(int(counter/N*100)) + '%')
        counter += 1
        
        try:
            tmp = pdr.get_data_yahoo(listassets[i], start=start_date, end=end_date)[feature]
            assets[listassets[i]] = tmp
        except:
            print(listassets[i], " could not be imported.")

    return assets



def convert_multicol_df_tolist(df, start_date, end_date):
    """
    Converts the multi-columns data frame obtained from get_assets_from_yahoo() into a list of timeseries.
    """
    # Forming a list of timeseries
    list_ts = []
    shared_index = df.index
    for c in df.columns:
        tmp_df = pd.DataFrame(data=df[start_date:end_date][c], index=shared_index)
        list_ts.append(ts.timeseries(tmp_df, name=c))
    
    return list_ts



def get_marketcap_today(market):
    """
    Function returning the market capitalization as it is today.
    """
    marketcap_today = pdr.data.get_quote_yahoo(market.columns)['marketCap']
    marketcap = pd.Series(data=marketcap_today, index=marketcap_today.index)
    return marketcap


def market_EWindex(market):
    """
    Sums all assets to make an index, corresponds to the EW portfolio.
    
    The formula for the weights here is:
    w_i = c / N for all i
    and we choose c = N so that \sum_i w_i = N.
    Thus we get the value at time t of the whole portfolio:
    M_t = \sum_i w_i m_{ti} = \sum_i m_{ti}
    """
    
    market_index = pd.DataFrame(market.sum(axis=1), columns=["Market EW Index"])
    return market_index


def market_CWindex(market, marketcap):
    """
    Function that returns the cap-weighted portfolio associated with the assets of a market.
    We compute the total return at time t, called R_t as:
    R_t = \sum_{i=1}^N v_i^t r_i^t / (\sum_{j=1}^N v_j^t)
    And since we only have data of the v_j's today we compute it is:
    R_t = \sum_{i=1}^N v_i^today r_i^t / (\sum_{j=1}^N v_j^today)
    
    Arguments:
    - market: the assets composing the market
    - marketvol: the volatility of these assets
    """
    
    # Initialization
    if (set(market.columns) != set(marketcap.index)):
        print(market.columns)
        print(marketcap.index)
        raise Exception("Error: the two data sources need to have same columns.")
    Nassets = market.shape[1]
    
    # Computing weighted returns
    M = Nassets * (market * marketcap / marketcap.sum()).sum(axis=1)
    market_index = pd.DataFrame(data=M, index=market.index, columns=["Market CW Index (Cap from last day)"])
    
    return market_index






