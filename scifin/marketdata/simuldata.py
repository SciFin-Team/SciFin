# Created on 2020/7/22

# This module is for simulating market data.

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import random as random
import matplotlib.pyplot as plt

from . import marketdata

#---------#---------#---------#---------#---------#---------#---------#---------#---------#


# GENERAL FUNCTIONS RELATED TO MARKET

def create_market(r_ini=100.0, drift=0.07, sigma=0.15, n_years=10,
                  steps_per_year=12, n_scenarios=1000):
    """
    Creates a market from a Geometric Brownian process for each stock.
    
    The model is of the form:
    r_t = drift * dt + sigma * \sqrt(dt) * \eps_t
    where r_t is the return series, mu is a drift (annualized),
    sigma is the volatility (annualised).
    
    Parameters
    ----------
    r_ini : float
      Initial value of the stock.
    drift : float
      Value of the drift.
    sigma : float
      Volatility of the process.
    n_years : int
      Number of years to generate.
    step_per_year : int
      Number of steps per year.
    n_scenarios : int
      Number of scenarios.
    
    Returns
    -------
    DataFrame
      Data frame of returns for the market.
    """
    
    # Checks
    assert(isinstance(n_years, int))
    assert(isinstance(steps_per_year, int))
    assert(isinstance(n_scenarios, int))
    
    # Initialization
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    
    # Computing r_t + 1
    rets_plus_1 = np.random.normal(loc=(1+drift)**dt,
                                   scale=(sigma*np.sqrt(dt)),
                                   size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    market_returns = r_ini * pd.DataFrame(rets_plus_1).cumprod()
    
    return market_returns


def set_market_names(data, date, date_type="end", interval_type='D'):
    """
    Sets the column and row names of the market dataframe.
      
    Parameters
    ----------
    data : DataFrame
      Dataframe on which we want to apply the function.
    date : str
      A specific date.
    date_type : str
      Value "end" for 'date' specifying the data end date, "start" for the start date.
    interval_type : str or DateOffset
      Specifies nature of the jump between two dates ('D' for days, 'M' for months, 'Y' for years).
    
    Returns
    -------
    None
      None
    
    Raises
    ------
      ValueError if the choice for 'date_type' is neither "start" or "end".
    
    Notes
    -----
     The two ways ("end" and "start") of specifying the dates are approximative.
     Uncertainty on the dates are of the order of the interval type.
     
     For offset aliases available see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Initializations
    n_ticks = data.shape[0]
    n_assets = data.shape[1]
    
    # Setting the column names
    data.columns = map(lambda x: "Asset " + str(x), range(n_assets))
    
    # Setting the row names
    # Quick check the current date has the right format:
    try:
        date = datetime.strptime(date, "%Y-%m-%d")
    except:
        ValueError("Current date format does not seem right.")
        
    # Generate the dates
    # either from end date
    if date_type == "start":
        
        if interval_type == 'D':
            date_series = date + pd.to_timedelta(np.arange(n_ticks), unit='D')
            
        elif interval_type == 'M':
            date_series = date + pd.to_timedelta(np.arange(n_ticks) * 12, unit='D')
            
        elif interval_type == 'Y':
            date_series = date + pd.to_timedelta(np.arange(n_ticks) * 365, unit='D')
            
    # or from the start date
    elif date_type == "end":
        
        if interval_type == 'D':
            date_series = date - timedelta(days=n_ticks) \
                               + pd.to_timedelta(np.arange(n_ticks), unit='D')
            
        elif interval_type == 'M':
            date_series = date - timedelta(days=int(n_ticks * (365./12.))) \
                               + pd.to_timedelta(np.arange(n_ticks) * int(365./12.), unit='D')
            
        elif interval_type == 'Y':
            date_series = date - timedelta(days=int(n_ticks * 365)) \
                               + pd.to_timedelta(np.arange(n_ticks) * 365, unit='D')
            
    else:
        ValueError("date_type choice is not recognized.")
        
    # Affecting the value to the rows names
    data.index = date_series.to_period(interval_type)
    
    return None


def is_index_valid(market):
    """
    Checks if the market has a correct index, meaning no date value is repeated.
    
    Parameters
    ----------
    market : DataFrame
      The market to be used.
    
    Returns
    -------
    bool
      Returns True if the index is valid, False otherwise.
    """
    
    index = market.index.tolist()
    market_set = set(index)
    
    for s in market_set:
        if index.count(s) > 1:
            return False
    return True
    

def create_market_shares(market, mean=100000, stdv=10000):
    """
    Creates a list of randomly generated numbers of shares for a market.
    The number of shares is generated from a normal distribution.
    
    Parameters
    ----------
    market : DataFrame
      The market we want to create shares for.
    mean : float
      The average value of a market share.
    stdv : float
      The standard deviation of the market shares.
      
    Returns
    -------
    DataFrame
      The data frame containing the market shares.
    """
    
    # number of shares we want
    n_assets = market.shape[1]
    
    market_shares = pd.Series( [int(np.random.normal(loc=mean, scale=stdv, size=1)) 
                                for _ in range(n_assets)] )
    market_shares.index = market.columns
    
    if market_shares.min() < 0:
        raise Exception("A negative market share was generated, please launch again.")
    
    return market_shares


def plot_market_components(market, dims=(10,5), legend=True):
    """
    Plots the assets contribution to the Equally-Weighted (EW) index.
    
    Parameters
    ----------
    market : DataFrame
      The market we take values from.
    dims : (int,int)
      Dimensions of the plot.
    legend : bool
      Option to plot the legend from market names.
    
    Returns
    -------
    None
      None
    """
    
    # Computing the EW portfolio
    market_EW = marketdata.market_EWindex(market)

    # Plotting market
    axis = market_EW.plot(figsize=dims, color='k', lw=3, legend=legend)
    
    # Plotting individual portfolios
    x = market.index.values.tolist()
    y = market.to_numpy().transpose()

    # Stack plot
    axis.stackplot(x, y, labels=market.columns.tolist())
    if legend:
        axis.legend(loc='upper left')

    return None

        

# FUNCTIONS USED WITH GENETIC ALGORITHM

def propagate_individual(individual, environment, name_indiv="Portfolio"):
    """
    Propagates the initial individual over time by computing its sum of gene values.
    
    Parameters
    ----------
    individual : list of floats
      List of Ngenes elements that represent our initial individual.
    environment : DataFrame
      Describes the time evolution of genes composing the individual.
    name_indiv : str
      Name of the individual.
    
    Returns
    -------
    DataFrame
      Pandas data frame containing the sum value of genes.
    
    Notes
    -----
      In the context of portfolios, an individual would be a portfolio of assets,
      Ngenes would be the number of assets in it, environment would be the market
      that leads the changes in asset values.
    """
    
    # Checks
    first_row = environment.iloc[0]
    is_uniform = True
    first_value = first_row[0]
    for x in first_row:
        if x != first_value:
            raise Error("First row of environment must be uniform in value.")
    
    # Initializations
    Ngenes = len(individual)
    
    # Propagating individuals
    portfolio = environment / first_value * individual
    
    # Summing contributions
    portfolio_total = pd.DataFrame(portfolio.sum(axis=1), columns=[name_indiv])
    
    return portfolio_total


def evaluation_dates(environment, n_dates=10, interval_type='M'):
    """
    Produces a number of equally spaced dates
    at which the individuals will be evaluated.
    
    Parameters
    ----------
    environment : DataFrame
      Represents the environment, i.e. the time evolution of gene values.
    n_dates : int
      Number of evaluation dates to generate.
    interval_type : str or Offset string
      Type of time interval between dates.
    
    Returns
    -------
    PeriodIndex
      List of dates used for evaluation.
    
    Notes
    -----
      In the context of portfolios, an individual would be a portfolio of assets,
      environment would be the market that leads the changes in asset values.
      
      To learn more about pandas .to_period() function, please refer to:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.to_period.html
    """
    
    # Checks
    assert(n_dates)
    
    # Initialization
    n_ticks = environment.shape[0]
    indices = np.linspace(start = 0, stop = n_ticks-1, num = n_dates+1).astype('int')
    
    # Find the corresponding dates
    special_dates = environment.index.to_timestamp()[indices].to_period(interval_type)
    
    # Raising exceptions if generated dates aren't satisfactory
    if special_dates[0] != environment.index[0]:
        raise Exception("ERROR !")
    if special_dates[-1] != environment.index[-1]:
        raise Exception("ERROR !")
    
    return special_dates


def find_tick_before_eval(environment_dates, eval_date):
    """
    Returns the tick before the evaluation date.
    
    Parameters
    ----------
    environment_dates : List of Period dates
      List of the dates given in the environment.
    eval_date : Period date
      Evaluation date for which we want the previous environment date.
    
    Returns
    -------
    Period date
      Environment date just before the evaluation date.
    """
    
    # Checks
    if (eval_date in environment_dates) == False:
        raise Exception("It appears that eval_date does not belong to the environment dates.")
    
    # Returning value
    for d in environment_dates:
        if d == eval_date:
            return d-1
    raise Exception("No date was found.")
    
    
def limited_propagation(population, environment, start, end):
    """
    Propagates the population over time, like `propagate_individual`,
    but only for a limited period of time and several individuals.

    Parameters
    ----------
    population : DataFrame
      Population made of different individuals.
    environment : DataFrame
      Represents the environment, i.e. the time evolution of gene values.
    start : Period date
      Starting date for the evolution.
    end : Period date
      Ending date of the evolution.
      
    Returns
    -------
    DataFrame
      Pandas data frame containing the individuals of the populations as columns
      and whose evolution is given along the time index.
    
    Notes
    -----
      In the context of portfolios, an individual would be a portfolio of assets,
      environment would be the market that leads the changes in asset values.
      
      To learn more about pandas .to_period() function, please refer to:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.to_period.html
    """
    
    # Initialization
    n_indiv = population.shape[0]
    n_genes = environment.shape[1]
    
    # Propagating individuals and adding them to a data frame
    list_portfolios = pd.DataFrame()
    
    for x in range(n_indiv):
        portfolio_name = population.index[x]
        
        # Computing (price of asset) x (asset allocation)
        portfolio = environment[start:end] * population.iloc[x][:n_genes] \
                                           * ( n_genes / population.iloc[x][:n_genes].sum())
        list_portfolios[portfolio_name] = portfolio.sum(axis=1)

    return pd.DataFrame(list_portfolios)


def portfolio_vol(weights, cov_matrix):
    """
    Returns the volatility of a portfolio from a covariance matrix and weights.
    
    Parameters
    ----------
    weights : Numpy Array
      Array of size N or matrix of size N x 1.
    cov_matrix : Numpy Array
      Matrix of size N x N.
      
    Returns
    -------
    float
      Volatility associated to the weights and covariance matrix.
    """
    return (weights.T @ cov_matrix @ weights)**0.5


def fitness_calculation(population, propagation, environment, current_eval_date, next_eval_date,
                        lamb=0.5, fitness_method="Last Return and Vol"):
    """
    Computes the fitness of each individual in a population.
    
    Different fitness methods can be used:
    - "Last Return": uses the value of return at the end of propagation,
                     i.e. the value just before evaluation date.
    - "Last Return and Vol": combines last return with estimation of volatility,
                             the dosage between the two being tuned by lambda.
    - "Avg Return and Vol": uses average return during the propagation and the
                            estimation of volatility, also using lambda.
    - "Sharpe Ratio": uses the ratio of average return over estimation of volatility.
    
    Parameters
    ----------
    population : DataFrame
      Population to evolve.
    propagation : DataFrame
      Time evolution of individuals.
    environment : DataFrame
      Environment which serves as a basis for propagation.
    current_eval_date : Period date
      Present date on which we evaluate the individuals.
    next_eval_date : Period date
      Target date until which we want to make the portfolios evolve.
    lamb : float in [0,1]
      Parameter of the fitness calculation to decide between returns and volatility.
    fitness_method : str
      Name of the fitness method chosen among:
      {"Last Return", "Last Return and Vol", "Avg Return and Vol", "Sharpe Ratio"}
    
    Returns
    -------
    List of floats
      List of fitness values for each individual of the population.
    
    Raises
    ------
    Exception
      In case the entered fitness_method name is not known.
    
    Notes
    -----
     Population has rows which are the names of the individuals (e.g. portfolios)
     and columns which are the genes (e.g. assets).
    
     Propagation has rows which are the time stamps,
     and columns which are the names of the individuals (e.g. portfolios).
    
    Examples
    --------
      None
    """
        
    # Method of last return
    if fitness_method == "Last Return":
        fitness_value = [propagation[x][-1] for x in propagation.columns]
        
        
    # Method combining last return and average volatility
    elif fitness_method == "Last Return and Vol":
        # Computing fitness from returns,
        # taking the last row value of each columns (i.e. each portfolio)
        fitness_from_return = [propagation[x][-1] for x in propagation.columns]

        # Defining the environment (i.e. market) correlation over a period of time
        # (here it does not really matter which one)
        covmat = environment.loc[current_eval_date : next_eval_date].corr()

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
        fitness_value = [ lamb * normalized_fitness_from_return[x] 
                          + (1-lamb) / normalized_fitness_from_vol[x]  
                          for x in range(len(fitness_from_return)) ]
    
    
    # Method combining average return and average volatility
    elif fitness_method == "Avg Return and Vol":
        # Computing fitness from returns,
        # taking the last row value of each columns (i.e. each portfolio)
        fitness_from_return = [ propagation[x].pct_change()[1:].mean()
                                for x in propagation.columns ]

        # Defining the environment (i.e. market) correlation over a period of time
        # (here it does not really matter which one)
        covmat = environment.loc[current_eval_date : next_eval_date].corr()

        # Loop over portfolios
        pop = population.filter(regex="Asset")
        fitness_from_vol = []
        for x in propagation.columns:
            # Taking the weights for an output portfolio
            weights = pop.loc[x]
            # Computing fitness from volatility
            fitness_from_vol.append(portfolio_vol(weights, covmat))

        # Combining the 2 fitnesses
        fitness_value = [ lamb * fitness_from_return[x]
                               + (1-lamb) / fitness_from_vol[x] 
                          for x in range(len(fitness_from_return)) ]
    
    
    # Method based on the Sharpe Ratio
    # We assume the risk-free rate is 0% to avoid introducing an arbitrary value here.
    elif fitness_method == "Sharpe Ratio":
        # Computing fitness from returns,
        # taking the last row value of each columns (i.e. each portfolio)
        fitness_from_return = [ propagation[x].pct_change()[1:].mean()
                                for x in propagation.columns ]

        # Defining the environment correlation over a period of time
        # (here it does not really matter which one)
        covmat = environment.loc[current_eval_date : next_eval_date].corr()

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
        raise Exception("Specified fitness method does not exist.")
    
    return fitness_value



# VISUALIZATION METHODS

def visualize_portfolios_1(market, list_individuals, evaluation_dates,
                           dims=(10,5), xlim=None, ylim=None):
    """
    Allows a quick visualization of the market,
    some sparse individuals, and the evaluation dates.
    
    Parameters
    ----------
    market : DataFrame
      Market from which we extract data about genes (i.e. assets)
    list_individuals : DataFrame
      Propagation of individuals over time.
    evaluation_dates : List of Period dates
      Dates at which we want to evaluate the individuals.
    dims : (float, float)
      (Optional) Dimensions of the plot.
    xlim : (float, float)
      (Optional) Range in x.
    ylin : (float, float)
      (Optional) Range in y.
    
    Returns
    -------
    None
      None
    """
    
    # Computing the EW portfolio
    market_EW = marketdata.market_EWindex(market)

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
    
    return None


def visualize_portfolios_2(market, marketcap, list_individuals, evaluation_dates,
                           dims=(10,5), xlim=None, ylim=None, savefile=False,
                           namefile="Result.png"):
    """
    Allows a quick visualization of market,
    some sparse individuals, and the evaluation dates.
    
    Parameters
    ----------
    market : DataFrame
      Market from which we extract data about assets (i.e. genes).
    marketcap : Panda.Series
      Market capitalization of the assets.
    list_individuals : DataFrame
      Propagation of individuals over time.
    evaluation_dates : List of Period dates
      Dates at which we want to evaluate the individuals.
    dims : (float, float)
      (Optional) Dimensions of the plot.
    xlim : (float, float)
      (Optional) Range in x.
    ylin : (float, float)
      (Optional) Range in y.
    savefile : bool
      Option to save the plot.
    namefile : str
      Name of the file to save in.
    
    Returns
    -------
    None
      None
    """
    
    # Initialization
    fig, axis = plt.subplots(nrows=1, ncols=1)
    
    # Computing the EW portfolio
    market_EW = marketdata.market_EWindex(market)
    market_CW = marketdata.market_CWindex(market, marketcap)

    # Plotting market
    market_EW.plot(figsize=dims, color='black',
                   linestyle='--', linewidth=1,
                   ax=axis, legend=False)
    
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
    market_EW.plot(figsize=dims, color='black', linestyle='--', linewidth=1, ax=axis)

    # Plotting the Cap-Weighted index so that it appears on top
    market_CW.plot(figsize=dims, color='black', linestyle='-', linewidth=1, ax=axis)
    
    # Set axes limits
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)
        
    return None


def show_allocation_distrib(step, saved_gens, eval_dates, n_bins=50,
                            savefile=False, namefile="Allocation_Distribution.png"):
    """
    Plots the distribution of saved generations (including elites and individuals)
    for a certain step of the loop that ran in `Genetic_Portfolio_Routine`.
    
    Since there are different individuals, we sum over these elements
    for each asset and we divide by the number of individuals.
    
    Parameters
    ----------
    step : int
      Step of the loop.
    saved_gens : DataFrame
      Generations to plot from.
    eval_dates : List of Period dates
      Evaluation dates for display.
    n_bins : int
      Number of bins.
    savefile : bool
      Option to save the plot.
    namefile : str
      Name of the file to save in.
    
    Returns
    -------
    None
      None
    """
    
    # Checks
    assert(isinstance(step, int))
    assert(isinstance(n_bins, int))
    
    # Initialization
    Nloops = len(saved_gens)-1
    tmp = (saved_gens[step].sum() / saved_gens[step].shape[0]).tolist()
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(15,5))

    # Assuming the largest allocations are always at the last step (which may not be true)
    xmin = min((saved_gens[Nloops].sum() / saved_gens[Nloops].shape[0]).tolist()) * 1.2
    xmax = max((saved_gens[Nloops].sum() / saved_gens[Nloops].shape[0]).tolist()) * 1.2

    # Plotting
    plt.hist(x=tmp, bins=n_bins, range=[xmin,xmax])
    plt.title("Histogram of allocations - "
              + eval_dates[step].to_timestamp().strftime("%Y-%m-%d"))
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)

    return None


def config_4n(n, market, VIX, savefile=False, namefile="VIX_derived_quantities.png"):
    """
    Plots the evaluation dates, ndays, mutation rate and fitness lambda
    as computed from the Volatility Index (VIX) and 4 structure numbers.
    
    Evaluation dates are computed according to the VIX, trying to get more evaluations
    when the VIX takes high values.
    
    This function is used to set the configuration of a dynamical creation
    of evaluation dates and mutation rate.
    
    Parameters
    ----------
    n : 4-tuple of int
      Structure parameters.
    market : DataFrame
      Market from which we extract data about assets (i.e. genes).
    VIX : DataFrame
      Values of the VIX over time.
    savefile : bool
      Option to save the plot.
    namefile : str
      Name of the file to save in.
      
    Returns
    -------
    None
      None
    """
    
    # Checks
    assert(len(n)==4)
    
    # Initializations
    n1, n2, n3, n4 = n
    market_dates = market.index.to_timestamp().strftime("%Y-%m-%d").tolist()
    save_eval_dates = []
    save_mutation_rate = []
    save_ndays = []
    save_fitness_lambda = []

    # Loop
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
        
        # Getting the VIX
        VIX_tmp = VIX[eval_date.to_timestamp().strftime("%Y-%m-%d")]
        
        # Computing the mutation rate
        ng_mutation_rate = (1 + (VIX_tmp/10).astype('int')) * (market.shape[1] / n3)
        save_mutation_rate.append(ng_mutation_rate)
        
        # Computing the fitness lambda
        VIX_past_max = VIX[:eval_date.to_timestamp().strftime("%Y-%m-%d")].max()
        fitness_lambda = 1 - VIX_tmp / (VIX_past_max * n4)
        save_fitness_lambda.append(100 * fitness_lambda)
        
        # Loop counter update
        loop +=1
        
        
    # Converting interesting quantities into a data frames
    df_mutation_rate = pd.DataFrame(data=save_mutation_rate,
                                    index=save_eval_dates,
                                    columns=["Mutation Rate"])
    
    df_ndays = pd.DataFrame(data=save_ndays,
                            index=save_eval_dates,
                            columns=["ndays"])
    
    df_fitness_lambda = pd.DataFrame(data=save_fitness_lambda,
                                     index=save_eval_dates,
                                     columns=["100 x Fitness lambda"])
    

    # Plotting quantities
    fig, axis = plt.subplots(nrows=1, ncols=1)
    VIX.plot(label="VIX")
    df_ndays.plot(figsize=(25,5), ax=axis, color="orange", linewidth=2, legend=True)
    df_mutation_rate.plot(legend=True, ax=axis, color="green", linewidth=2)
    df_fitness_lambda.plot(legend=True, ax=axis, color="purple", linewidth=2)
    
    # Plotting y=0 line and evaluation dates
    axis.axhline(0, color='red', linestyle='--', linewidth=2)
    for ed in save_eval_dates:
        axis.axvline(x=ed.to_timestamp().strftime("%Y-%m-%d"), color='grey', linestyle='--')
    axis.legend()

    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)
        
    return None


def plot_diff_GenPort_CW(saved_propags, market_CW, eval_dates,
                         savefile=False, namefile="ResultDifference.png"):
    """
    Computes and plots the difference between the portfolios
    of the genetic algorithm and the Cap-Weighted Portfolio.
    
    Parameters
    ----------
    saved_propags : DataFrame
      Propagations that have been saved.
    market_CW : DataFrame
      Cap-Weighted portfolio to compare with.
    eval_dates : List of Period dates
      Evaluation dates for display.
    savefile : bool
      Option to save the plot.
    namefile : str
      Name of the file to save in.
      
    Returns
    -------
    None
      None
    """
    
    # Checks that all frequencies are the same
    if (saved_propags.index.freq != market_CW.index.freq):
        market_CW = market_CW.asfreq(saved_propags.index.freq)
    if (saved_propags.index.freq != eval_dates.freq):
        eval_dates = eval_dates.asfreq(saved_propags.index.freq)
    
    # Computing values
    diff_array = (saved_propags.to_numpy() - market_CW.to_numpy()) \
                        / market_CW.to_numpy() * 100
        
    Diff_GenPort_CW = pd.DataFrame(data = diff_array,
                                   columns = saved_propags.columns,
                                   index=saved_propags.index)

    # Plotting
    fig, axis = plt.subplots(nrows=1, ncols=1)
    Diff_GenPort_CW.plot(legend=False, figsize=(20,7), ax=axis)
    
    # Adding evaluation dates
    for ed in eval_dates:
        axis.axvline(x=ed, color='grey', linestyle='--', linewidth=1)
    
    # Setting axes
    axis.axhline(y=0, color='grey', linestyle='--')
    plt.title("Difference Genetic Portfolios - CW Portfolio")
    plt.xlabel("Time")
    plt.ylabel("Difference in %")
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)
    
    return None


def plot_asset_evol(n, eval_dates, saved_gens,
                    savefile=False, namefile="asset_evol.png"):
    """
    Plots the evolution of asset allocations over time.
    
    Parameters
    ----------
    n : int
      Number refering to the asset.
    eval_dates : List of Period dates
      Evaluation dates for display.
    saved_gens : DataFrame
      Generations to plot from.
    savefile : bool
      Option to save the plot.
    namefile : str
      Name of the file to save in.
      
    Returns
    -------
    None
    """
    
    # Checks
    assert(isinstance(n, int))
    
    # Forming the set of portfolio names
    set_indices = []
    for x in range(len(saved_gens)):
        set_indices = set(set_indices).union(set(saved_gens[x].index.tolist()))
    
    # Creating the empty data frame
    asset_evol = pd.DataFrame(data=None, columns=eval_dates, index=set_indices)

    # Computing
    for x in range(len(eval_dates)):
        colname = eval_dates[x].to_timestamp().strftime("%Y-%m-%d")
        for y in saved_gens[x].index:
            asset_evol.loc[y,colname] = saved_gens[x].loc[y,'Asset ' + str(n)]
    
    # Transpose
    asset_t = asset_evol.transpose()
    
    # Plot
    asset_t.plot(figsize=(15,5), legend=False)
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)

    return None


        
#---------#---------#---------#---------#---------#---------#---------#---------#---------#