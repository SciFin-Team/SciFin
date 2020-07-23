# Created on 2020/7/22

# This module is for simulating market data.

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import random as random
import matplotlib.pyplot as plt

import marketdata.marketdata as md


def create_market(r_ini=100.0, drift=0.07, sigma=0.15, n_years=10, steps_per_year=12, n_scenarios=1000):
    """
    Method that creates a market from a Geometric Brownian process for each stock of the form:
    r_t = drift * dt + sigma * \sqrt(dt) * \eps_t
    where r_t is the return series, my is a drift (annualized), sigma is the volatility (annualised)
    """
    
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    
    # Computing r_t + 1
    rets_plus_1 = np.random.normal(loc=(1+drift)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = r_ini * pd.DataFrame(rets_plus_1).cumprod()
    
    return ret_val



def set_market_names(data, date, date_type="end", interval_type='D'):
    """
    Function that sets the column and row names of the market dataframe.
    
    Arguments:
    - data: dataframe on which we want to apply the function
    - date: a specific date
    - date_type: "end" for date specifying the end date of the data, "start" for the start date
    - interval_type: specifies what jumps correspond to ('D' for days, 'M' for months, 'Y' for years)
    
    Note: the two ways ("end" and "start") of specifying the dates are approximative.
          The uncertainty on the dates are of the order of the interval type.
    """
    
    Nticks = data.shape[0]
    Nassets = data.shape[1]
    
    # Setting the column names
    data.columns = map(lambda x: "Asset " + str(x), range(Nassets))
    
    # Setting the row names
    # Quick check the current date has the right format:
    try:
        date = datetime.strptime(date, "%Y-%m-%d")
    except:
        ValueError("Current date format does not seem right.")
        
    # Generate the dates, either from end date or start date
    if date_type == "start":
        if interval_type == 'D':
            date_series = date + pd.to_timedelta(np.arange(Nticks), unit='D')
        elif interval_type == 'M':
            date_series = date + pd.to_timedelta(np.arange(Nticks) * 12, unit='D')
        elif interval_type == 'Y':
            date_series = date + pd.to_timedelta(np.arange(Nticks) * 365, unit='D')
    elif date_type == "end":
        if interval_type == 'D':
            date_series = date - timedelta(days=Nticks) + pd.to_timedelta(np.arange(Nticks), unit='D')
        elif interval_type == 'M':
            date_series = date - timedelta(days=int(Nticks * (365./12.))) + pd.to_timedelta(np.arange(Nticks) * int(365./12.), unit='D')
        elif interval_type == 'Y':
            date_series = date - timedelta(days=int(Nticks * 365)) + pd.to_timedelta(np.arange(Nticks) * 365, unit='D') 
    else:
        ValueError("date_type choice is not recognized.")
        
    # Affecting the value to the rows names
    data.index = date_series.to_period(interval_type)
    return



def is_index_valid(market):
    """
    Checks if the market has a correct index, meaning no date value is repeated.
    """
    index = market.index.tolist()
    market_set = set(index)
    
    is_valid = True
    for s in market_set:
        if index.count(s) > 1:
            return False
    

def create_market_shares(market, mean=100000, stdv=10000):
    """
    Function that creates a list of randomly generated numbers of shares
    
    Arguments:
    - market: the market we want to create shares for
    - mean: the average value of a market share
    - stdv: the standard deviation
    """
    
    # number of shares we want
    Nassets = market.shape[1]
    
    market_shares = pd.Series([int(np.random.normal(loc=mean, scale=stdv, size=1)) for _ in range(Nassets)])
    market_shares.index = market.columns
    
    if market_shares.min() < 0:
        raise Exception("A negative market share was generated, please launch again.")
    
    return market_shares



def plot_market_components(market, dims=(10,5), legend=True):
    """
    Function that plots the assets contribution to the EW index
    """
    
    # Computing the EW portfolio
    market_EW = md.market_EWindex(market)

    # Plotting market
    axis = market_EW.plot(figsize=dims, legend=legend)
    
    # Plotting individual portfolios
    x = market.index.values.tolist()
    # y = np.array([market[c].values.tolist() for c in market.columns])
    y = market.to_numpy().transpose()

    axis.stackplot(x, y, labels=market.columns.tolist())
    if legend:
        axis.legend(loc='upper left')



        

# FUNCTIONS USED WITH GENETIC ALGORITHM

def propagate_investments(investment, market, name_indiv="Portfolio"):
    """
    Function that propagates the initial investments into a portfolio over time.
    
    Argument:
    - individual: that's a list of Nassets elements that represent our initial investment.
    - market: the market (set of assets) on which the investments are applied.
    - name_indiv: name of the individual portfolio.
    """
    
    # Check
    first_row = market.iloc[0]
    is_uniform = True
    first_value = first_row[0]
    for x in first_row:
        if x != first_value:
            raise Error("First row of market must be uniform in value.")
    
    Nassets = len(investment)
    
    # Propagating investments
    portfolio = market / first_value * investment
    
    # Summing contributions
    portfolio_total = pd.DataFrame(portfolio.sum(axis=1), columns=[name_indiv])
    return portfolio_total


