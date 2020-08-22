# Created on 2020/7/22

# This module is for simulating market data.

# Standard library imports
from datetime import datetime
from datetime import timedelta
import random as random

# Third party imports
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt

# Local application imports
from . import marketdata
from .. import timeseries


# Dictionary of Pandas' Offset Aliases
# and their numbers of appearance in a year.
DPOA = {'D': 365, 'B': 252, 'W': 52,
        'SM': 24, 'SMS': 24, 
        'BM': 12, 'BMS': 12, 'M': 12, 'MS': 12,
        'BQ': 4, 'BQS': 4, 'Q': 4, 'QS': 4,
        'Y': 1, 'A':1}

# Datetimes format
fmt = "%Y-%m-%d %H:%M:%S"
fmtz = "%Y-%m-%d %H:%M:%S %Z%z"


#---------#---------#---------#---------#---------#---------#---------#---------#---------#

# CLASS FOR MARKET

class Market:
    """
    Creates a market.
    
    Attributes
    ----------
    data : DataFrame
      Contains a time-like index and columns of values for each market component.
    start_utc : Pandas.Timestamp
      Starting date.
    end_utc : Pandas.Timestamp
      Ending date.
    dims : 2-tuple (int,int)
      Dimensions of the market data.
    freq : str or None
      Frequency inferred from index.
    name : str
      Name or nickname of the market.
    tz : str
      Timezone name.
    timezone : pytz timezone
      Timezone associated with dates.
    units : List of str
      Unit of the market data columns.
    """
    
    def __init__(self, df=None, tz=None, units=None, name=""):

        # Deal with DataFrame
        if (df is None) or (df.empty == True):
            self.data = pd.DataFrame(index=None, data=None)
            self.start_utc = None
            self.end_utc = None
            self.dims = (0,0)
            self.freq = None
            self.name = 'Empty Market'
        else:         
            # Extract values
            if type(df.index[0]) == 'str':
                new_index = pd.to_datetime(df.index, format=fmt)
                self.data = pd.DataFrame(index=new_index, data=df.values)
                self.start_utc = datetime.strptime(str(new_index[0]), fmt)
                self.end_utc = datetime.strptime(str(new_index[-1]), fmt)
                self.dims = df.shape
                try:
                    self.freq = pd.infer_freq(new_index)
                except:
                    self.freq = 'Unknown'
                self.name = name
            else:
                self.data = df
                self.start_utc = df.index[0]
                self.end_utc = df.index[-1]
                self.dims = df.shape
                try:
                    self.freq = pd.infer_freq(df.index)
                except:
                    self.freq = 'Unknown'
                self.name = name
                
        # Deal with unit     
        if units is None:
            self.units = None
        else:
            assert(len(units) == len(self.data.columns))
            self.units = units
        
        # Deal with timezone
        if tz is None:
            self.tz = 'UTC'
            self.timezone = pytz.utc
        else:
            self.tz = tz
            self.timezone = pytz.timezone(tz)


    def is_index_valid(self):
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

        index = self.data.index.tolist()
        market_set = set(index)

        for s in market_set:
            if index.count(s) > 1:
                return False
        return True
    
    
    def reset_index(self, new_index):
        """
        Resets the index with a new one given in argument.
        """
        
        # Checks
        try:
            assert(len(new_index) == self.data.shape[0])
        except AssertionError:
            AssertionEror("New index should have same dimension as current index.")
    
        # Replacing index
        self.data.index = new_index
        
        return None


    def to_list(self, start_date=0, end_date=-1):
        """
        Converts the Market data frame into a list of TimeSeries.

        Parameters
        ----------
        market : Market
          Market to convert.
        start_date : str or datetime
          Starting date we want for the time series.
        end_date : str or datetime
          Ending date we want for the time series.

        Returns
        -------
        List of TimeSeries
          The list of times series extracted from the data frame.
        """

        # Forming a list of timeseries
        list_ts = []
        shared_index = self.data.index

        # Loop over columns
        for c in self.data.columns:
            tmp_df = pd.DataFrame(data=self.data[start_date:end_date][c], index=shared_index)
            list_ts.append(timeseries.TimeSeries(tmp_df, name=c))

        return list_ts


    
    
# GENERAL FUNCTIONS RELATED TO MARKET


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
    ValueError
      If the choice for 'date_type' is neither "start" or "end".
    
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


def create_market_returns(r_ini, drift, sigma, n_years,
                          steps_per_year, n_components,
                          date, date_type, interval_type='D',
                          tz=None, units=None, name=""):
    """
    Creates a market from a Geometric Brownian process for each stock.
    
    The model for each stock is of the form:
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
    n_components : int
      Number of components of the market.
    
    Notes
    -----
      All stocks are assumed to be in the same time zone.
    
    Returns
    -------
    Market
      Market of returns for the market.
    """
    
    # Checks
    assert(isinstance(n_years, int))
    assert(isinstance(steps_per_year, int))
    assert(isinstance(n_components, int))
    
    # Initialization
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    
    # Compute r_t + 1
    rets_plus_1 = np.random.normal(loc=(1+drift)**dt,
                                   scale=(sigma*np.sqrt(dt)),
                                   size=(n_steps, n_components))
    rets_plus_1[0] = 1
    df_returns = r_ini * pd.DataFrame(rets_plus_1).cumprod()

    # Set market index and column names
    set_market_names(df_returns, date=date, date_type=date_type, interval_type=interval_type)
    
    # Make a market
    market_returns = Market(df=df_returns, tz=tz, units=units, name=name)
    
    return market_returns


def create_market_shares(market, mean=100000, stdv=10000):
    """
    Creates a list of randomly generated numbers of shares for a market.
    The number of shares is generated from a normal distribution.
    
    Parameters
    ----------
    market : Market
      The market we want to create shares for.
    mean : float
      The average value of a market share.
    stdv : float
      The standard deviation of the market shares.
      
    Returns
    -------
    Pandas Series
      The pandas series containing the market shares.
    """
    
    # Checks
    assert(isinstance(market, Market))
    
    # Get number of assets
    n_assets = market.data.shape[1]
    
    # Create market shares
    market_shares = pd.Series( [int(np.random.normal(loc=mean, scale=stdv, size=1)) 
                                for _ in range(n_assets)] )
    market_shares.index = market.data.columns
    
    # Checks
    if market_shares.min() < 0:
        raise ValueError("A negative market share was generated, please launch again.")
    
    return market_shares


# VISUALIZATION METHODS

def plot_market_components(market, dims=(10,5), legend=True):
    """
    Plots the assets contribution to the Equally-Weighted (EW) index.
    
    Parameters
    ----------
    market : Market
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
    x = market.data.index.values.tolist()
    y = market.data.to_numpy().transpose()

    # Stack plot
    axis.stackplot(x, y, labels=market.data.columns.tolist())
    if legend:
        axis.legend(loc='upper left')

    return None

        
    
    
# FUNCTIONS USED WITH GENETIC ALGORITHM

def propagate_individual(individual, environment, name_indiv="Portfolio"):
    """
    Propagates the initial individual over time by computing its sum of gene values.

    The series of values is then stored in the attribute Individual.history through
    a pandas DataFrame.
    
    Parameters
    ----------
    individual : Individual
      Individual whose genes will be used.
    environment : Market
      Describes the time evolution of genes composing the individual.
    name_indiv : str
      Name of the individual.
    
    Returns
    -------
    None
      None
    
    Notes
    -----
      In the context of portfolios, an individual would be a portfolio of assets,
      Ngenes would be the number of assets in it, environment would be the market
      that leads the changes in asset values.
    """
    
    # Checks
    first_row = environment.data.iloc[0]
    is_uniform = True
    first_value = first_row[0]
    for x in first_row:
        if x != first_value:
            raise ValueError("First row of environment must be uniform in value.")
    
    # Initializations
    Ngenes = individual.ngenes
    
    # Propagating individuals
    portfolio = environment.data / first_value * individual.genes
    
    # Summing contributions
    individual.history = pd.DataFrame(portfolio.sum(axis=1), columns=[name_indiv])
    
    return None


def evaluation_dates(environment, n_dates=10, interval_type='M'):
    """
    Produces a number of equally spaced dates
    at which the individuals will be evaluated.
    
    Parameters
    ----------
    environment : Market
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
    n_ticks = environment.dims[0]
    indices = np.linspace(start = 0, stop = n_ticks-1, num = n_dates+1).astype('int')
    
    # Find the corresponding dates
    special_dates = environment.data.index.to_timestamp()[indices].to_period(interval_type)
    
    # Raise exceptions if generated dates aren't satisfactory
    if special_dates[0] != environment.data.index[0]:
        raise IndexError("Generated dates unsatisfactory !")
    if special_dates[-1] != environment.data.index[-1]:
        raise IndexError("Generated dates unsatisfactory !")
    
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
        raise ValueError("It appears that eval_date does not belong to the environment dates.")
    
    # Returning value
    for d in environment_dates:
        if d == eval_date:
            return d-1
    raise ValueError("No date was found.")
    
    
def limited_propagation(population, environment, start, end):
    """
    Propagates the population over time, like `propagate_individual`,
    but only for a limited period of time and several individuals.

    It stores a pandas DataFrame in the attribute Population.history,
    this DataFrame contains the individuals of the populations as columns
    and their evolution is given along the time index.
    
    Parameters
    ----------
    population : Population
      Population made of different individuals.
    environment : Market
      Represents the environment, i.e. the time evolution of gene values.
    start : Period date
      Starting date for the evolution.
    end : Period date
      Ending date of the evolution.
      
    Returns
    -------
    DataFrame
      
    
    Notes
    -----
      In the context of portfolios, an individual would be a portfolio of assets,
      environment would be the market that leads the changes in asset values.
      
      To learn more about pandas .to_period() function, please refer to:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.to_period.html
    """
    
    # Initialization
    n_indiv = population.n_indiv
    n_genes = population.n_genes
    
    # Propagating individuals and adding them to a data frame
    list_portfolios = pd.DataFrame()
    
    for x in range(n_indiv):
        portfolio_name = population.data.index[x]
        
        # Computing (price of asset) x (asset allocation)
        portfolio = environment.data[start:end] * population.data.iloc[x][:n_genes] \
                                                * ( n_genes / population.data.iloc[x][:n_genes].sum())
        list_portfolios[portfolio_name] = portfolio.sum(axis=1)

    # Store the data frame into Population.history
    population.history = pd.DataFrame(list_portfolios)
        
    return None


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


def fitness_calculation(population, environment, current_eval_date, next_eval_date,
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
    population : Population
      Population to evolve.
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
    ValueError
      In case the entered fitness_method name is not known.
    
    Notes
    -----
     Population has rows which are the names of the individuals (e.g. portfolios)
     and columns which are the genes (e.g. assets).
    
     The propagation, i.e. population.history, has rows which are the time stamps,
     and columns which are the names of the individuals (e.g. portfolios).
    
    Examples
    --------
      None
    """
        
    # Method of last return
    if fitness_method == "Last Return":
        fitness_value = [population.history[x][-1] for x in population.history.columns]
        
        
    # Method combining last return and average volatility
    elif fitness_method == "Last Return and Vol":
        # Computing fitness from returns,
        # taking the last row value of each columns (i.e. each portfolio)
        fitness_from_return = [population.history[x][-1] for x in population.history.columns]

        # Defining the environment (i.e. market) correlation over a period of time
        # (here it does not really matter which one)
        covmat = environment.data.loc[current_eval_date : next_eval_date].corr()

        # Loop over portfolios
        pop = population.data.filter(regex="Asset")
        fitness_from_vol = []
        for x in population.history.columns:
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
        fitness_from_return = [ population.history[x].pct_change()[1:].mean()
                                for x in population.history.columns ]

        # Defining the environment (i.e. market) correlation over a period of time
        # (here it does not really matter which one)
        covmat = environment.data.loc[current_eval_date : next_eval_date].corr()

        # Loop over portfolios
        pop = population.data.filter(regex="Asset")
        fitness_from_vol = []
        for x in population.history.columns:
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
        fitness_from_return = [ population.history[x].pct_change()[1:].mean()
                                for x in population.history.columns ]

        # Defining the environment correlation over a period of time
        # (here it does not really matter which one)
        covmat = environment.data.loc[current_eval_date : next_eval_date].corr()

        # Loop over portfolios
        pop = population.data.filter(regex="Asset")
        fitness_from_vol = []
        for x in population.history.columns:
            # Taking the weights for an output portfolio
            weights = pop.loc[x]
            # Computing fitness from volatility
            fitness_from_vol.append(portfolio_vol(weights, covmat))

        # Combining the 2 fitnesses
        fitness_value = [fitness_from_return[x] / fitness_from_vol[x]  for x in range(len(fitness_from_return))]
        
        
    # Otherwise return Exception
    else:
        raise ValueError("Specified fitness method does not exist.")
    
    return fitness_value



# VISUALIZATION METHODS

def visualize_portfolios_1(market, propagation, evaluation_dates,
                           dims=(10,5), xlim=None, ylim=None):
    """
    Allows a quick visualization of the market,
    some sparse individuals, and the evaluation dates.
    
    Parameters
    ----------
    market : Market
      Market from which we extract data about genes (i.e. assets)
    propagation : DataFrame
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
    for name in propagation.columns:
        propagation[name].plot(ax=axis)
        
    # Plotting evaluation dates
    for ed in evaluation_dates:
        axis.axvline(x=ed, color='grey', linestyle='--')
    
    # Set axes limits
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    
    return None


def visualize_portfolios_2(market, marketcap, propagation, evaluation_dates,
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
    propagation : DataFrame
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
    for name in propagation.columns:
        propagation[name].plot(ax=axis)

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
    nloops = len(saved_gens)-1
    tmp = (saved_gens[step].sum() / saved_gens[step].shape[0]).tolist()
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(15,5))

    # Assuming the largest allocations are always at the last step (which may not be true)
    xmin = min((saved_gens[nloops].sum() / saved_gens[nloops].shape[0]).tolist()) * 1.2
    xmax = max((saved_gens[nloops].sum() / saved_gens[nloops].shape[0]).tolist()) * 1.2

    # Plotting
    plt.hist(x=tmp, bins=n_bins, range=[xmin,xmax])
    plt.title("Histogram of allocations - "
              + eval_dates[step].to_timestamp().strftime("%Y-%m-%d"))
    
    # Saving plot as a png file
    if savefile:
        plt.savefig('./' + namefile)

    return None


def config_4n(n, market, vix, savefile=False, namefile="VIX_derived_quantities.png"):
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
    market : Market
      Market from which we extract data about assets (i.e. genes).
    vix : DataFrame
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
    market_dates = market.data.index.to_timestamp().strftime("%Y-%m-%d").tolist()
    save_eval_dates = []
    save_mutation_rate = []
    save_ndays = []
    save_fitness_lambda = []

    # Loop
    loop = 0
    eval_date = market.data.index[0]
    next_eval_date = market.data.index[10]

    while next_eval_date < market.data.index[-1]:

        # Updating
        eval_date = next_eval_date
        save_eval_dates.append(next_eval_date)

        # Computing the number of days to next date
        vix_ateval = 1 + (vix[eval_date.to_timestamp().strftime("%Y-%m-%d")]/n1).astype('int')
        ndays = n2 - vix_ateval
        save_ndays.append(ndays)

        if ndays <= 0:
            raise ValueError("Distance between dates must be strictly positive !")

        # Computing next date of evaluation
        current_date_index = market_dates.index(eval_date.to_timestamp().strftime("%Y-%m-%d"))
        if current_date_index + ndays < market.data.shape[0]:
            next_eval_date = market.data.index[current_date_index + ndays]
        else:
            next_eval_date = market.data.index[-1]
        
        # Getting the VIX
        vix_tmp = vix[eval_date.to_timestamp().strftime("%Y-%m-%d")]
        
        # Computing the mutation rate
        ng_mutation_rate = (1 + (vix_tmp/10).astype('int')) * (market.data.shape[1] / n3)
        save_mutation_rate.append(ng_mutation_rate)
        
        # Computing the fitness lambda
        vix_past_max = vix[:eval_date.to_timestamp().strftime("%Y-%m-%d")].max()
        fitness_lambda = 1 - vix_tmp / (vix_past_max * n4)
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
    vix.plot(label="VIX")
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