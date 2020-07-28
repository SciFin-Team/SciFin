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
    
    for s in market_set:
        if index.count(s) > 1:
            return False
    return True
    

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



def evaluation_dates(market, Ndates=10, interval_type='M'):
    """
    Function producing a number of equally spaced dates at which the portfolios will be evaluated.
    
    Arguments:
    - market: the dataframe representing the market (assets values over time)
    - Ndates: the number of dates
    """
    
    # Initialization
    Nticks = market.shape[0]
    indices = np.linspace(start=0, stop=Nticks-1, num=Ndates+1).astype('int')
    
    # Find the corresponding dates
    special_dates = market.index.to_timestamp()[indices].to_period(interval_type)
    
    # Check
    if special_dates[0] != market.index[0]:
        raise Exception("ERROR !")
    if special_dates[-1] != market.index[-1]:
        raise Exception("ERROR !")
    
    return special_dates



def find_tick_before_eval(market_dates, date):
    """
    Function returning the tick before the evaluation date.
    """
    
    # Check:
    if (date in market_dates) == False:
        raise Exception("It appears that the date does not belong to the market dates.")
    
    # Returning value
    for d in market_dates:
        if d == date:
            return d-1
    raise Exception("Apparently no date was found.")
    
    

def limited_propagation(population, market, start, end):
    """
    Function that propagates the initial investments into a portfolio over time, like `propagate_investments`, but only for a limited period of time.
    Also, the function is extended from the case of one individual portfolio to a dataframe of them.
    
    Argument:
    - individual: that's a list of Nassets elements that represent our initial investment
    - market: the market (set of assets) on which the investments are applied
    - start: starting date or period
    - end: ending date or period
    - name_indiv: name of the individual portfolio
    """
    
    # Initialization
    Nindiv = population.shape[0]
    Nassets = market.shape[1]
    
    # Propagating investments, we will do a list of dataframes
    list_portfolios = pd.DataFrame()
    for x in range(Nindiv):
        portfolio_name = population.index[x]
        # Computing (price of asset) x (asset allocation)
        # portfolio = market[start:end] / 100 * population.iloc[x]
        portfolio = market[start:end] * population.iloc[x][:Nassets] * ( Nassets / population.iloc[x][:Nassets].sum())
        list_portfolios[portfolio_name] = portfolio.sum(axis=1)

    return pd.DataFrame(list_portfolios)



def portfolio_vol(weights, cov_matrix):
    """
    Function returning the volatility of a portfolio from a covariance matrix and weights.
    weights are a numpy array or N x 1 matrix and covmat is an N x N matrix.
    """
    vol = (weights.T @ cov_matrix @ weights)**0.5
    return vol



def fitness_calculation(population, propagation, market, current_eval_date, next_eval_date, lamb=0.5, fitness_method="Max Return and Vol"):
    """
    Function that simply collects the last value in time of each portfolio and consider it as the fitness measure.
    
    Note: - population has rows which are the names of the portfolio, and columns which are the assets.
          - propagation has rows which are the time stamps, and columns which are the names of the portfolios.
    """
        
    # Method of max return
    if fitness_method == "Max Return":
        fitness_value = [propagation[x][-1] for x in propagation.columns]
        
        
    # Method combining max return and average volatility
    elif fitness_method == "Max Return and Vol":
        # Computing fitness from returns, taking the last row value of each columns (i.e. each portfolio)
        fitness_from_return = [propagation[x][-1] for x in propagation.columns]

        # Defining the market correlation over a period of time (here it does not really matter which one)
        covmat = market.loc[current_eval_date : next_eval_date].corr()

        # Loop over portfolios
        pop = population.filter(regex="Asset")
        fitness_from_vol = []
        for x in propagation.columns:
            # Taking the weights for an output portfolio
            weights = pop.loc[x]
            # Computing fitness from volatility
            fitness_from_vol.append(portfolio_vol(weights, covmat))

        # Normalizing
        normalized_fitness_from_return = fitness_from_return / sum(fitness_from_return)
        normalized_fitness_from_vol = fitness_from_vol / sum(fitness_from_vol)

        # Combining the 2 fitnesses
        fitness_value = [lamb * normalized_fitness_from_return[x] + (1-lamb) / normalized_fitness_from_vol[x]  for x in range(len(fitness_from_return))]
    
    
    # Method combining average return and average volatility
    elif fitness_method == "Avg Return and Vol":
        # Computing fitness from returns, taking the last row value of each columns (i.e. each portfolio)
        fitness_from_return = [propagation[x].pct_change()[1:].mean() for x in propagation.columns]

        # Defining the market correlation over a period of time (here it does not really matter which one)
        covmat = market.loc[current_eval_date : next_eval_date].corr()

        # Loop over portfolios
        pop = population.filter(regex="Asset")
        fitness_from_vol = []
        for x in propagation.columns:
            # Taking the weights for an output portfolio
            weights = pop.loc[x]
            # Computing fitness from volatility
            fitness_from_vol.append(portfolio_vol(weights, covmat))

        # Combining the 2 fitnesses
        fitness_value = [lamb * fitness_from_return[x] + (1-lamb) / fitness_from_vol[x]  for x in range(len(fitness_from_return))]
    
    
    # Method based on the Sharpe Ratio - We assume the risk-free rate is 0% to avoid introducing an arbitrary value here.
    elif fitness_method == "Sharpe Ratio":
        # Computing fitness from returns, taking the last row value of each columns (i.e. each portfolio)
        fitness_from_return = [propagation[x].pct_change()[1:].mean() for x in propagation.columns]

        # Defining the market correlation over a period of time (here it does not really matter which one)
        covmat = market.loc[current_eval_date : next_eval_date].corr()

        # Loop over portfolios
        pop = population.filter(regex="Asset")
        fitness_from_vol = []
        for x in propagation.columns:
            # Taking the weights for an output portfolio
            weights = pop.loc[x]
            # Computing fitness from volatility
            fitness_from_vol.append(portfolio_vol(weights, covmat))

        # Combining the 2 fitnesses
        fitness_value = [fitness_from_return[x] / fitness_from_vol[x]  for x in range(len(fitness_from_return))]
        
        
    # Otherwise return Exception
    else:
        raise Exception("Specified fitness method does not seem to exist.")
    
    
    return fitness_value





def visualize_portfolios_1(market, list_individuals, evaluation_dates, dims=(10,5), xlim=None, ylim=None):
    """
    Function that allows a quick visualization of market, some sparse individuals, and the evaluation dates
    """
    
    # Computing the EW portfolio
    market_EW = md.market_EWindex(market)

    # Plotting market
    axis = market_EW.plot(figsize=dims)
    
    # Plotting individual portfolios
    for name in list_individuals.columns:
        list_individuals[name].plot(ax=axis)
        
    # Plotting evaluation dates
    for ed in evaluation_dates:
        axis.axvline(x=ed, color='grey', linestyle='--')
    
    # Set axes limits
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    
    return



def visualize_portfolios_2(market, marketcap, list_individuals, evaluation_dates, dims=(10,5), xlim=None, ylim=None, savefile=False, namefile="Result.png"):
    """
    Function that allows a quick visualization of market, some sparse individuals, and the evaluation dates
    """
    
    # Initialization
    fig, axis = plt.subplots(nrows=1, ncols=1)
    
    # Computing the EW portfolio
    market_EW = md.market_EWindex(market)
    market_CW = md.market_CWindex(market, marketcap)

    # Plotting market
    market_EW.plot(figsize=dims, color='black', linestyle='--', linewidth=1, ax=axis, legend=False)
    
    # Plotting evaluation dates
    for ed in evaluation_dates:
        axis.axvline(x=ed, color='grey', linestyle='--', linewidth=1)
    
    # Computing the PW portfolio
    # market_shares = market.iloc[0]
    # market_PW = market_PWindex(market, market_shares)
    # market_PW.plot(ax=axis, color='k', linestyle='-', linewidth=2)
    
    # Plotting individual portfolios
    for name in list_individuals.columns:
        list_individuals[name].plot(ax=axis)

    # Re-Plotting market to appear on top
    #axis = market_EW.plot(figsize=dims, color='gray', linestyle='-', linewidth=1)
    market_EW.plot(figsize=dims, color='black', linestyle='--', linewidth=1, ax=axis)

    # Plotting the Cap-Weighted index so that it appears on top
    market_CW.plot(figsize=dims, color='black', linestyle='-', linewidth=1, ax=axis)
    
    # Set axes limits
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)


        
        
def show_allocation_distrib(step, save_gens, save_eval_dates, Nbins=50, savefile=False, namefile="Allocation_Distribution.png"):
    """
    Plots the distribution of a generation (including elites and individuals) for a certain step of the loop that ran in Genetic_Portfolio_Routine.
    Since there are different individuals, we sum over these elements for each asset, and we divide by the number of individuals.
    """
    
    # Initialization
    Nloops = len(save_gens)-1
    tmp = (save_gens[step].sum() / save_gens[step].shape[0]).tolist()
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(15,5))

    # Assuming the largest allocations are always at the last step (which may not be true)
    xmin = min((save_gens[Nloops].sum() / save_gens[Nloops].shape[0]).tolist()) * 1.2
    xmax = max((save_gens[Nloops].sum() / save_gens[Nloops].shape[0]).tolist()) * 1.2

    # Plotting
    plt.hist(x=tmp, bins=Nbins, range=[xmin,xmax])
    plt.title("Histogram of allocations - " + save_eval_dates[step].to_timestamp().strftime("%Y-%m-%d"))
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)







def config_4n(n1,n2,n3,n4, market, VIX, savefile=False, namefile="VIX_derived_quantities.png"):
    """
    Function that plots the evaluation dates, ndays, mutation rate and fitness lambda from the VIX and 4 structure numbers.
    It also mimicks the general loop, so that we can see how dates are evaluated and how quantities are computed.
    """

    market_dates = market.index.to_timestamp().strftime("%Y-%m-%d").tolist()
    
    save_eval_dates = []
    save_mutation_rate = []
    save_ndays = []
    save_fitness_lambda = []

    loop = 0
    eval_date = market.index[0]
    next_eval_date = market.index[10]


    while next_eval_date < market.index[-1]:

        # Updating
        eval_date = next_eval_date
        save_eval_dates.append(next_eval_date)

        # Computing the number of days to next date
        VIX_ateval = 1 + (VIX[eval_date.to_timestamp().strftime("%Y-%m-%d")]/n1).astype('int')
        ndays = n2 - VIX_ateval
        save_ndays.append(ndays)

        if ndays <= 0:
            raise Exception("Distance between dates must be strictly positive !")

        # Computing next date of evaluation
        current_date_index = market_dates.index(eval_date.to_timestamp().strftime("%Y-%m-%d"))
        if current_date_index + ndays < market.shape[0]:
            next_eval_date = market.index[current_date_index + ndays]
        else:
            next_eval_date = market.index[-1]

        # Computing the mutation rate
        ng_mutation_rate = (1 + (VIX[eval_date.to_timestamp().strftime("%Y-%m-%d")]/10).astype('int')) * (market.shape[1] / n3) # Big change here!
        save_mutation_rate.append(ng_mutation_rate)
        
        # Computing the fitness lambda
        fitness_lambda = 1 - VIX[eval_date.to_timestamp().strftime("%Y-%m-%d")] / (VIX[:eval_date.to_timestamp().strftime("%Y-%m-%d")].max() * n4)
        save_fitness_lambda.append(100 * fitness_lambda)
        
        # Loop counter update
        loop +=1
        
        
    # Converting the mutation rate into a dataFrame
    df_mutation_rate = pd.DataFrame(data=save_mutation_rate, index=save_eval_dates, columns=["Mutation Rate"])
    df_ndays = pd.DataFrame(data=save_ndays, index=save_eval_dates, columns=["ndays"])
    df_fitness_lambda = pd.DataFrame(data=save_fitness_lambda, index=save_eval_dates, columns=["100 x Fitness lambda"])

    # PLOTTING EVALUATION DATES / MUTATION RATES
    fig, axis = plt.subplots(nrows=1, ncols=1)
    VIX.plot(label="VIX")
    df_ndays.plot(figsize=(25,5), ax=axis, color="orange", linewidth=2, legend=True)
    df_mutation_rate.plot(legend=True, ax=axis, color="green", linewidth=2)
    df_fitness_lambda.plot(legend=True, ax=axis, color="purple", linewidth=2)
    axis.axhline(0, color='red', linestyle='--', linewidth=2)
    for ed in save_eval_dates:
        axis.axvline(x=ed.to_timestamp().strftime("%Y-%m-%d"), color='grey', linestyle='--')
    axis.legend()

    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)
        
        
        
        
def plot_diff_GenPort_CW(save_propags, market_CW, evaluation_dates, savefile=False, namefile="ResultDifference.png"):
    """
    Computes and plot the difference between the portfolios of the genetic algorithm and the Cap-Weighted Portfolio.
    """
    
    # Computing values - Old way (very slow)
    # Diff_GenPort_CW = save_propags.copy()
    # for i in range(Diff_GenPort_CW.shape[0]):
    #     market = market_CW.values[i][0]
    #     for j in range(Diff_GenPort_CW.shape[1]):
    #         Diff_GenPort_CW.iloc[i][j] = (Diff_GenPort_CW.iloc[i][j] - market) / market * 100
    
    # Computing values - New way (much much faster !)
    diff_array = (save_propags.to_numpy() - market_CW.to_numpy()) / market_CW.to_numpy() * 100
    Diff_GenPort_CW = pd.DataFrame(diff_array, columns = save_propags.columns, index=save_propags.index)

    # Plotting
    fig, axis = plt.subplots(nrows=1, ncols=1)
    Diff_GenPort_CW.plot(legend=False, figsize=(20,7), ax=axis)
    
    # Adding evaluation dates
    for ed in evaluation_dates:
        axis.axvline(x=ed, color='grey', linestyle='--', linewidth=1)
    
    # Setting axes
    axis.axhline(y=0, color='grey', linestyle='--')
    plt.title("Difference Genetic Portfolios - CW Portfolio")
    plt.xlabel("Time")
    plt.ylabel("Difference in %")
    
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)
    
    return



def plot_asset_evol(n, save_eval_dates, save_gens, savefile=False, namefile="asset_evol.png"):

    # Forming the set of portfolio names
    set_indices = []
    for x in range(len(save_gens)):
        set_indices = set(set_indices).union(set(save_gens[x].index.tolist()))
    
    # Creating the empty data frame
    asset_evol = pd.DataFrame(data=None, columns=save_eval_dates, index=set_indices)

    # Computing
    for x in range(len(save_eval_dates)):
        colname = save_eval_dates[x].to_timestamp().strftime("%Y-%m-%d")
        for y in save_gens[x].index:
            asset_evol.loc[y,colname] = save_gens[x].loc[y,'Asset ' + str(n)]
    
    # Transpose
    asset_t = asset_evol.transpose()
    
    # Plot
    asset_t.plot(figsize=(15,5), legend=False)
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)
        
        
        