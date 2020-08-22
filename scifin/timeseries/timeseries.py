# Created on 2020/7/15

# This module is for the class TimeSeries and related functions.

# Standard library imports
from datetime import datetime

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Local application imports
# /


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


def get_list_timezones():
    """
    Lists all the time zone names that can be used.
    """
    print(pytz.all_timezones)
    return None
        


# CLASS Series

class Series:
    """
    Abstract class defining a Series and its methods.
    
    This class serves as a parent class for TimeSeries and CatTimeSeries.
    
    Attributes
    ----------
    data : DataFrame
      Contains a time-like index and for each time a single value.
    start_utc : Pandas.Timestamp
      Starting date.
    end_utc : Pandas.Timestamp
      Ending date.
    nvalues : int
      Number of values, i.e. also of dates.
    freq : str or None
      Frequency inferred from index.
    name : str
      Name or nickname of the series.
    unit : str or None
      Unit of the series values.
    tz : str
      Timezone name.
    timezone : pytz timezone
      Timezone associated with dates.
    """
    
    def __init__(self, df=None, tz=None, unit=None, name=""):
        """
        Receives a data frame as an argument and initializes the time series.
        """
        
        # Deal with DataFrame
        if (df is None) or (df.empty == True):
            self.data = pd.DataFrame(index=None, data=None)
            self.start_utc = None
            self.end_utc = None
            self.nvalues = 0
            self.freq = None
            self.name = 'Empty TimeSeries'
        else:
            # Making sure the dataframe is just
            # an index + 1 value column
            try:
                assert(df.shape[1]==1)
            except AssertionError:
                raise AssertionError("Time series must be built from a data frame with only one value column.")
                         
            # Extract values
            if type(df.index[0]) == 'str':
                new_index = pd.to_datetime(df.index, format=fmt)
                self.data = pd.DataFrame(index=new_index, data=df.values)
                self.start_utc = datetime.strptime(str(new_index[0]), fmt)
                self.end_utc = datetime.strptime(str(new_index[-1]), fmt)
                self.nvalues = df.shape[0]
                try:
                    self.freq = pd.infer_freq(self.data.index)
                except:
                    self.freq = 'Unknown'
                self.name = name
            else:
                self.data = df
                self.start_utc = df.index[0]
                self.end_utc = df.index[-1]
                self.nvalues = df.shape[0]
                try:
                    self.freq = pd.infer_freq(self.data.index)
                except:
                    self.freq = 'Unknown'
                self.name = name
                
        # Deal with unit     
        if unit is None:
            self.unit = None
        else:
            self.unit = unit
        
        # Deal with timezone
        if tz is None:
            self.tz = 'UTC'
            self.timezone = pytz.utc
        else:
            self.tz = tz
            self.timezone = pytz.timezone(tz)
        
        
    def get_start_date_local(self):
        """
        Returns the attribute UTC start date in local time zone defined by attribute timezine.
        """
        start_tmp = datetime.strptime(str(self.start_utc), fmt).astimezone(self.timezone)
        return datetime.strftime(start_tmp, format=fmtz)
    
    
    def get_end_date_local(self):
        """
        Returns the attribute UTC end date in local time zone defined by attribute timezine.
        """
        end_tmp = datetime.strptime(str(self.end_utc), fmt).astimezone(self.timezone)
        return datetime.strftime(end_tmp, format=fmtz)

    
    def specify_data(self, start, end):
        """
        Returns the appropriate data according to user's specifying
        or not the desired start and end dates.
        """
         
        # Prepare data frame
        if (start is None) and (end is None):
            data = self.data

        elif (start is None) and (end is not None):
            data = self.data[:end]

        elif (start is not None) and (end is None):
            data = self.data[start:]

        elif (start is not None) and (end is not None):
            data = self.data[start:end]

        return data

    
    def start_end_names(self, start, end):
        """
        Recasts the time series dates to 10 characters strings
        if the date hasn't been re-specified (i.e. value is 'None').
        """
        s = str(self.start_utc)[:10] if (start is None) else start
        e = str(self.end_utc)[:10] if (end is None) else end

        return s,e
    
    
    def is_sampling_uniform(self):
        """
        Tests if the sampling of a time series is uniform or not.
        Returns a boolean value True when the sampling is uniform, False otherwise.
        """
        # Prepare data
        sampling = [datetime.timestamp(x) for x in self.data.index]
        assert(len(sampling)==self.nvalues)
        intervals = [sampling[x] - sampling[x-1] for x in range(1,self.nvalues,1)]

        # Testing
        prev = intervals[0]
        for i in range(1,len(intervals),1):
            if intervals[i] - prev > 1.e-6:
                return False
        return True
    

    


#---------#---------#---------#---------#---------#---------#---------#---------#---------#

# CLASS TimeSeries

class TimeSeries(Series):
    """
    Class defining a time series and its methods.
    
    This class inherits from the parent class 'Series'.
    
    Attributes
    ----------
    data : DataFrame
      Contains a time-like index and for each time a single value.
    start_utc : Pandas.Timestamp
      Starting date.
    end_utc : Pandas.Timestamp
      Ending date.
    nvalues : int
      Number of values, i.e. also of dates.
    freq : str or None
      Frequency inferred from index.
    name : str
      Name or nickname of the series.
    tz : str
      Timezone name.
    timezone : pytz timezone
      Timezone associated with dates.
    type : str
      Type of the series.
    unit : str or None
      Unit of the time series values.
    """
    
    def __init__(self, df=None, tz=None, unit=None, name=""):
        """
        Receives a data frame as an argument and initializes the time series.
        """

        super().__init__(df=df, tz=tz, unit=unit, name=name)
        
        # Add attributes initialization if needed
        self.type = 'TimeSeries'
    
    
    
    ### Plot INFORMATION ABOUT THE TIME SERIES ###
    
    def simple_plot(self, figsize=(12,5), dpi=100):
        """
        Plots the time series in a simple way.

        Parameters
        ----------
        figsize : 2-tuple of ints
          Dimensions of the figure.
        dpi : int
          Dots-per-inch definition of the figure.
          
        Returns
        -------
        None
          None
        """
        
        # Plot
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.data.index, self.data.values, color='k')
        
        # Make it cute
        if self.name is None:
            tmp_name = " "
        else:
            tmp_name = self.name
        title = "Time series " + tmp_name + " from " + str(self.start_utc)[:10] \
                + " to " + str(self.end_utc)[:10]
        if self.unit is None:
            ylabel = 'Value'
        else:
            ylabel = 'Value (' + self.unit + ')'
        plt.gca().set(title=title, xlabel="Date", ylabel=ylabel)
        plt.show()
        
        return None
    
    
    def distribution(self, start=None, end=None, bins=20, figsize=(8,4), dpi=100):
        """
        Plots the distribution of values between two dates.
        """
        
        # Prepare data frame
        data = self.specify_data(start, end)
            
        # Plot distribution of values
        plt.figure(figsize=figsize, dpi=dpi)
        data.hist(bins=bins, grid=False, color='w', lw=2, edgecolor='k')
        
        # Make it cute        
        s,e = self.start_end_names(start, end)
        title = "Distribution of values between " + s + " and " + e
        plt.gca().set(title=title, xlabel="Value", ylabel="Hits")
        plt.show()
        
        return None
        
        
    def density(self, start=None, end=None, bins=20, figsize=(8,4), dpi=100):
        """
        Plots the density of values between two dates.
        """
        
        # Prepare data frame
        data = self.specify_data(start, end)
        s,e = self.start_end_names(start, end)
        
        # Plot distribution of values
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        data.plot.density(color='k', ax=ax, legend=False)
        
        # Make it cute
        title = "Density plot of values between " + s + " and " + e
        plt.gca().set(title=title, xlabel="Value", ylabel="Density")
        plt.show()
        
        return None
    

    def simple_plot_distrib(self, start=None, end=None, bins=20, figsize=(10,4), dpi=100):
        """
        Plots the time series and its associated distribution of values between two dates.
        """

        # Checks
        assert(isinstance(bins,int))
        
        # Prepare data frame
        data = self.specify_data(start, end)
        s,e = self.start_end_names(start, end)
        
        # Plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(1, 4)
        
        # Plot 1 - Time Series simple plot
        f_ax1 = fig.add_subplot(gs[:, 0:3])
        f_ax1.plot(data.index, data.values, color='k')
        if self.name is None:
            tmp_name = " "
        else:
            tmp_name = self.name
        title1 = "Time series " + tmp_name + " from " + s + " to " + e
        if self.unit is None:
            ylabel = 'Value'
        else:
            ylabel = 'Value (' + self.unit + ')'
        plt.gca().set(title=title1, xlabel="Date", ylabel=ylabel)
        
        # Plot 2 - Distribution of values
        f_ax2 = fig.add_subplot(gs[:, 3:])
        data.hist(bins=bins, grid=False, ax=f_ax2, orientation="horizontal", color='w', lw=2, edgecolor='k')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0)
        title2 = "Distribution"
        plt.gca().set(title=title2, xlabel="Value", ylabel="Hits")
        
        return None
    
    
    def get_sampling_interval(self):
        """
        Returns the sampling interval for a uniformly-sampled time series.
        """
        if(self.is_sampling_uniform()==False):
            raise SamplingError("Error: the time series is not uniformly sampled.")
        else:
            idx1 = self.data.index[1]
            idx0 = self.data.index[0]
            intv = datetime.timestamp(idx1) - datetime.timestamp(idx0)
            return intv
        

    def lag_plot(self, lag=1, figsize=(5,5), dpi=100, alpha=0.5):
        """
        Returns the scatter plot x_t v.s. x_{t-l}.
        """
        # Check
        try:
            assert(lag>0)
        except AssertionError:
            raise AssertionError("The lag must be an integer equal or more than 1.")
        
        # Do the plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        pd.plotting.lag_plot(self.data, lag=lag, c='black', alpha=alpha)
        
        # Set title
        if self.name is None:
            tmp_name = " "
        else:
            tmp_name = self.name
        title = "Lag plot of time series" + tmp_name
        plt.gca().set(title=title, xlabel="x(t)", ylabel="x(t+"+str(lag)+")")
        plt.show()
        
        return None
        
        
    def lag_plots(self, nlags=5, figsize=(10,10), dpi=100, alpha=0.5):
        """
        Returns a number of scatter plots x_t v.s. x_{t-l}
        where l is the lag value taken from [0,...,nlags].
        
        Notes
        -----
          It is required that nlags > 1.
        """
        # Check
        try:
            assert(nlags>1)
        except AssertionError:
            raise AssertionError("nlags must be an integer starting from 2.")
            
        # Rule for the number of rows/cols
        ncols = int(np.sqrt(nlags))
        if(nlags%ncols==0):
            nrows = nlags//ncols
        else:
            nrows = nlags//ncols + 1
        
        # Do the plots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                                 figsize=figsize, dpi=dpi)
        for i, ax in enumerate(axes.flatten()[:nlags]):
            pd.plotting.lag_plot(self.data, lag=i+1, ax=ax, c='black', alpha=alpha)
            ax.set_xlabel("x(t)")
            ax.set_ylabel("x(t+"+str(i+1)+")")
        
        # Set title
        if self.name is None:
            tmp_name = " "
        else:
            tmp_name = self.name
        title = "Multiple lag plots of time series " + tmp_name
        fig.suptitle(title, )
        plt.show()
        
        return None
    
    
    ### SIMPLE DATA EXTRACTION ON THE TIME SERIES ###
    
    def hist_avg(self, start=None, end=None):
        """
        Returns the historical average of the time series
        between two dates (default is the whole series).
        """
        data = self.specify_data(start, end)
        avg = data.values.mean()
        
        return avg
    
    
    def hist_std(self, start=None, end=None):
        """
        Returns the historical standard deviation of the time series
        between two dates (default is the whole series).
        """
        data = self.specify_data(start, end)
        std = data.values.std()
        
        return std
        
        
    def hist_variance(self, start=None, end=None):
        """
        Returns the historical variance of the time series
        between two dates (default is the whole series).
        """
        data = self.specify_data(start, end)
        var = data.values.var()
        
        return var
    
    
    def hist_skewness(self, start=None, end=None):
        """
        Returns the historical skew of the time series
        between two dates (default is the whole series).
        """
        data = self.specify_data(start, end)
        skew = stats.skew(data.values)[0]
        
        return skew
    
    
    def hist_kurtosis(self, start=None, end=None):
        """
        Returns the historical (Fisher) kurtosis of the time series
        between two dates (default is the whole series).
        """
        data = self.specify_data(start, end)
        kurt = stats.kurtosis(data.values, fisher=False)[0]
        
        return kurt
    
    
    def min(self, start=None, end=None):
        """
        Returns the minimum of the series.
        """
        data = self.specify_data(start, end)
        ts_min = data.values.min()
        
        return ts_min
    
    
    def max(self, start=None, end=None):
        """
        Returns the maximum of the series.
        """
        data = self.specify_data(start, end)
        ts_max = data.values.max()
        
        return ts_max
    
    
    def describe(self, start=None, end=None):
        """
        Returns description of time series between two dates.
        This uses the pandas function having same name.
        """
        data = self.specify_data(start, end)
        print(data.describe())
        return None
    
    
    ### METHODS THAT ARE CLOSER TO FINANCIAL APPLICATIONS ###
    
    def percent_change(self, start=None, end=None, name=""):
        """
        Returns the percent change of the series (in %).
        
        Notes
        -----
        When computing the percent change, first date gets
        NaN value and is thus removed from the time series.
        """
        data = self.specify_data(start, end)
        new_data = data.pct_change()
        new_ts = TimeSeries(new_data[1:], tz=self.tz, unit='%', name=name)
        
        return new_ts
    
    
    # Alias method of percent_change()
    # For people with a Finance terminology preference
    net_returns = percent_change
    
    
    def gross_returns(self, start=None, end=None, name=""):
        """
        Returns the gross returns of the series (in %),
        i.e. percent change + 1.
        
        Notes
        -----
        When computing the percent change, first date gets
        NaN value and is thus removed from the time series.
        """
        
        data = self.specify_data(start, end)
        new_data = 1 + data.pct_change()
        new_ts = TimeSeries(new_data[1:], tz=self.tz, name=name)
        
        return new_ts
    
    
    def hist_vol(self, start=None, end=None):
        """
        Computes the net returns of the time series and
        returns their associated historical volatility
        between two dates (default is the whole series).
        
        Notes
        -----
          When computing the percent change, first date gets
          NaN value and is thus removed from calculation.
        
          Since pandas.DataFrame.pct_change() returns values in
          percent, we divide by 100 to bring back numerical values.
        """
        
        # Initialization
        data = self.specify_data(start, end)
        
        # Warning message
        if self.is_sampling_uniform() is not True:
            print('Warning: Index not uniformly sampled. Result could be meaningless.')
            
        # Computing net returns
        net_returns = data.pct_change()[1:]
        
        # Compute standard deviation, i.e. volatility
        std = net_returns.values.std()
        
        return std
    
    
    def annualized_vol(self, start=None, end=None):
        """
        Returns the annualized volatility of the time series
        between two dates (default is the whole series),
        using the frequency of the time series when usable.
        """
        
        # Initializations
        hvol = self.hist_vol(start, end)
        
        if (self.freq is not None) and (self.freq in DPOA.keys()):
            return hvol * np.sqrt(DPOA[self.freq])
        else:
            raise ValueError('Annualized volatility could not be evaluated.')

    
    def annualized_return(self, start=None, end=None):
        """
        Returns the annualized return of the time series
        between two dates (default is the whole series),
        using the frequency of the time series when usable.
        """
        
        # Initializations
        gross_returns = self.gross_returns(start, end)
        prd = gross_returns.data.prod()[0]

        # Checks
        assert(gross_returns.nvalues == self.nvalues-1)
        if (gross_returns.freq != self.freq):
            print('Warning: gross_returns frequency and time series frequency do not match.')
            print('         In that context, results may not be making sense.')
        
        if (self.freq is not None) and (self.freq in DPOA.keys()):
            return prd**(DPOA[self.freq]/gross_returns.nvalues) - 1
        else:
            raise ValueError('Annualized return could not be evaluated.')
    
    
    def risk_ratio(self, start=None, end=None):
        """
        Returns the risk ratio, i.e. the ratio of annualized return
        over annualized volatility.
        """

        ann_return = self.annualized_return(start, end)
        ann_volatility = self.annualized_vol(start, end)
        
        return ann_return / ann_volatility

    
    def annualized_Sharpe_ratio(self, risk_free_rate=0, start=None, end=None):
        """
        Returns the Sharpe ratio, also known as risk adjusted return.
        """
        
        ann_return = self.annualized_return(start, end)
        ann_volatility = self.annualized_vol(start, end)
        
        return (ann_return - risk_free_rate) / ann_volatility
    
    
    
    ### METHODS RELATED TO VALUE AT RISK ###
    
    def hist_var(self, p, start=None, end=None):
        """
        Returns the historical p-VaR (Value at Risk) between two dates.
        
        Returns
        -------
        float
          VaR value computed between the chosen dates.
        """
        
        # Checks
        assert(p>=0 and p<=1)
        if 100*p%1 != 0:
            print("Warning: Probability too precise, only closest percentile computed here.")
            print("         Hence for p =", p, ", percentile estimation is based on p =", int(100*p), "%.")
        
        # Prepare data
        data = self.specify_data(start, end)
        
        return np.percentile(data.values, int(100*p))
    
    
    def hist_cvar(self, p, start=None, end=None):
        """
        Returns the historical CVaR (Conditional Value at Risk) between two dates.
        This quantity is also known as the Expected Shortfall (ES).
        
        Returns
        -------
        float
          CVaR value computed between the chosen dates.
        """
        
        # Checks
        assert(p>=0 and p<=1)
        if 100*p%1 != 0:
            print("Warning: Probability too precise, only closest percentile computed here.")
            print("         Hence for p =", p, ", percentile estimation is based on p =", int(100*p), "%.")
        
        # Prepare data
        data = self.specify_data(start, end)
        var = self.hist_var(p=p, start=start, end=end)
        
        # Computing CVaR
        tmp_sum = 0
        tmp_n = 0
        for val in data.values:
            if val <= var:
                tmp_sum += val[0]
                tmp_n += 1

        return tmp_sum / tmp_n

    
    # Alias method of hist_cvar
    # For people with a Finance terminology preference
    hist_expected_shortfall = hist_cvar
    
    
    def cornish_fisher_var(self, p, start=None, end=None):
        """
        Returns the VaR (Value at Risk) between two dates from
        the Cornish-Fisher expansion.
        
        Returns
        -------
        float
          VaR value computed between the chosen dates.
        """
        
        # Checks
        assert(p>=0 and p<=1)
        
        # Prepare data
        data = self.specify_data(start, end)

        # Compute z-score based on normal distribution
        z = stats.norm.ppf(p)
        
        # Compute modified z-score from expansion
        s = stats.skew(data.values)[0]
        k = stats.kurtosis(data.values, fisher=False)[0]
        new_z = z + (z**2 - 1) * s/6 + (z**3 - 3*z) * (k-3)/24 \
                  - (2*z**3 - 5*z) * (s**2)/36
        
        return data.values.mean() + new_z * data.values.std(ddof=0)
    
    
    
    ### AUTOCORRELATION COMPUTATION ###
    
    def autocorrelation(self, lag=1, start=None, end=None):
        """
        Returns the autocorrelation of the time series for a specified lag.
        
        We use the function:
        $rho_l = frac{Cov(x_t, x_{t-l})}{\sqrt(Var[x_t] Var[x_{t-l}])}
        where $x_t$ is the time series at time t.
        
        Cov denotes the covariance and Var the variance.
        
        We also use the properties $rho_0 = 1$ and $rho_{-l} = rho_l$
        (using LaTeX notations here).
        """
        
        # Initialization
        l = abs(lag)
        
        # Trivial case
        if l==0:
            return 1
        
        # Prepare data frame
        data = self.specify_data(start, end)
        
        # General case
        assert(l < data.shape[0])
        shifted_data = data.shift(l)
        numerator = np.mean((data - data.mean()) * (shifted_data - shifted_data.mean()))
        denominator = data.std() * shifted_data.std()
        
        return numerator / denominator
    
    
    def plot_autocorrelation(self, lag_min=0, lag_max=25, start=None, end=None,
                             figsize=(8,4), dpi=100):
        """
        Uses autocorrelation method in order to return a plot
        of the autocorrelation againts the lag values.
        """
        
        # Checks
        assert(lag_max>lag_min)
        
        # Computing autocorrelation
        x_range = list(range(lag_min, lag_max+1, 1))
        ac = [self.autocorrelation(lag=x, start=start, end=end) for x in x_range]
        
        # Plot
        plt.figure(figsize=figsize, dpi=dpi)
        plt.bar(x_range, ac, color='w', lw=2, edgecolor='k')
        s,e = self.start_end_names(start, end)
        title = "Autocorrelation from " + s + " to " + e + " for lags = [" \
                + str(lag_min) + "," + str(lag_max) + "]"
        plt.gca().set(title=title, xlabel="Lag", ylabel="Autocorrelation Value")
        plt.show()
        
        return None
        
    
    def acf_pacf(self, lag_max=25, figsize=(12,3), dpi=100):
        """
        Returns a plot of the AutoCorrelation Function (ACF)
        and Partial AutoCorrelation Function (PACF) from statsmodels.
        """
        # Plot
        fig, axes = plt.subplots(1,2, figsize=figsize, dpi=dpi)
        plot_acf(self.data.values.tolist(), lags=lag_max, ax=axes[0])
        plot_pacf(self.data.values.tolist(), lags=lag_max, ax=axes[1])
        plt.show()
        
        return None
    
    
    
    ### SIMPLE TRANSFORMATIONS OF THE TIME SERIES TO CREATE A NEW TIME SERIES ###
    
    def trim(self, new_start, new_end):
        """
        Method that trims the time series to the desired dates
        and send back a new time series.
        """
        new_df = self.data[new_start:new_end]
        new_ts = TimeSeries(new_df, tz=self.tz)
        
        return new_ts
    
    
    def add_cst(self, cst=0):
        """
        Method that adds a constant to the time series.
        """
        new_df = self.data + cst
        new_ts = TimeSeries(new_df, tz=self.tz)
        
        return new_ts
    
    
    def mult_by_cst(self, cst=1):
        """
        Method that multiplies the time series by a constant.
        """
        new_df = self.data * cst
        new_ts = TimeSeries(new_df, tz=self.tz)
        
        return new_ts
    
    
    def linear_combination(self, other_ts, factor1=1, factor2=1):
        """
        Method that adds a time series to the current one
        according to linear combination:
        factor1 * current_ts + factor2 * other_ts.
        """
        new_data = factor1 * np.array(self.data.values) + factor2 * np.array(other_ts.data.values)
        new_df = pd.DataFrame(index=self.data.index, data=new_data)
        new_ts = TimeSeries(new_df, tz=self.tz)
        
        return new_ts
    
    
    def convolve(self, func, x_min, x_max, n_points, normalize=False):
        """
        Performs a convolution of the time series with a function 'func'.
        The 'normalize' option allows to renormalize 'func' such that
        the sum of its values is one.
        
        Parameters
        ----------
        func : function
          Function we want to employ for convolution.
        x_min : float
          Minimum value to consider for 'func'.
        x_max : float
          Maximum value to consider for 'func'.
        n_points : int
          Number of points to consider in the function.
        normalize: bool
          Option to impose the sum of func values to be 1.
        """
        
        # Checks
        assert(isinstance(n_points, int))
        
        # Getting the time series values
        ts = self.data.values

        # Getting the convolving function values
        X = np.linspace(x_min, x_max, n_points)
        func_vals = []
        for x in X:
            func_vals.append(func(x))
        if normalize==True:
            sum_vals = np.array(func_vals).sum()
            func_vals /= sum_vals
        
        # Generate convolved values
        convolved_vals = np.convolve(func_vals, ts.flatten(), mode='same')
        convolved_ts = TimeSeries(pd.DataFrame(index=self.data.index, data=convolved_vals), tz=self.tz)
        
        return convolved_ts
    
    
    def get_drawdowns(self, start=None, end=None, name=""):
        """
        Computes the drawdowns and returns a new time series from them.
        
        Returns
        -------
        TimeSeries
          Time series of the drawdowns.
        """
        
        # Prepare data frame
        data = self.specify_data(start, end)
        
        # Compute drawdowns
        trailing_max = data.cummax()
        drawdowns = (data - trailing_max) / trailing_max
        
        # Make a time series from them
        new_ts = TimeSeries(drawdowns, tz=self.tz, name=name)
        
        return new_ts
    
    
    def max_drawdown(self, start=None, end=None, name=""):
        """
        Returns the maximum drawdown of a time series.
        
        Returns
        -------
        float
          Maximum drawdown.
        """
        
        # Prepare data frame
        data = self.specify_data(start, end)
        
        # Compute drawdowns
        trailing_max = data.cummax()
        drawdowns = (data - trailing_max) / trailing_max
        
        return -drawdowns.values.min()
    
    
    def divide_by_timeseries(self, other_ts, start=None, end=None, name=""):
        """
        Returns a time series from the division of the current time series
        with another time series (current_ts / other_ts).
        
        Returns
        -------
        TimeSeries
          Division time series.
        """
        
        # Prepare data frame
        data = self.specify_data(start, end)
        
        # Check that data has the same index
        # as the dividing time series
        assert(data.index.tolist() == other_ts.data.index.tolist())
        
        # Do the division
        new_data = np.array(data.values) / np.array(other_ts.data.values)
        new_df = pd.DataFrame(index=data.index, data=new_data)
        new_ts = TimeSeries(new_df, tz=self.tz, name=name)
        
        return new_ts
        
    
    def add_gaussian_noise(self, mu, sigma, start=None, end=None, name=""):
        """
        Adds a Gaussian noise to the current time series.
        
        Parameters
        ----------
        mu : float
          Mean parameter of the noise.
        sigma : float
          Standard deviation of the noise.
        start_utc : str
          Starting date.
        end_utc : str
          Ending date.
        name : str
          Name or nickname of the series.
        
        Returns
        -------
        TimeSeries
          Time series with added Gaussian noise.
        """
        
        # Prepare data frame
        data = self.specify_data(start, end)
        n = len(data.values)
        
        # Generate noise
        noise = np.random.normal(loc=mu, scale=sigma, size=n)
        
        # Generate new time series
        new_data = []
        for i in range(n):
            new_data.append(data.values[i][0] + noise[i])
        new_df = pd.DataFrame(index=data.index, data=new_data)
        new_ts = TimeSeries(new_df, tz=self.tz, name=name)
        
        return new_ts
    
    
    
    
    ### FITTING METHODS ###
    
    def rolling_avg(self, pts=1):
        """
        Transforms the time series into a rolling window average time series.
        """
        new_values = [self.data[x-pts+1:x].mean() for x in range(pts-1, self.nvalues, 1)]
        new_df = pd.DataFrame(index=self.data.index[pts-1:self.nvalues], data=new_values)
        new_ts = TimeSeries(new_df, tz=self.tz)
        
        return new_ts
    
    
    def polyfit(self, order=1, start=None, end=None):
        """
        Provides a polynomial fit of the time series.
        """
        
        # Prepar data
        data = self.specify_data(start, end)
        new_index = [datetime.timestamp(x) for x in data.index]
        new_values = [data.values.tolist()[x][0] for x in range(len(data))]
        
        # Do the fit
        fit_formula = np.polyfit(new_index, new_values, deg=order)
        model = np.poly1d(fit_formula)
        print("Evaluated model: \n", model)
        yfit = [model(x) for x in new_index]
        
        # Build data frame
        assert(len(data.index)==len(yfit))
        new_df = pd.DataFrame(index=data.index, data=yfit)
        new_ts = TimeSeries(new_df, tz=self.tz)
        
        return new_ts

    
    def sample_uniformly(self):
        """
        Returns a new time series for which the sampling is uniform.
        """
        
        # Check if actually we need to do something
        if self.is_sampling_uniform() == True:
            print("Time series already has a uniform sampling. \
                  Returning the same time series.")
            return self
        
        # Prepare the new index
        original_timestamps = [datetime.timestamp(x) for x in self.data.index]
        original_values = self.data.values
        N = len(original_values)
        assert(N>2)
        new_timestamps = np.linspace(original_timestamps[0], original_timestamps[-1], N)
        new_index = [datetime.fromtimestamp(x) for x in new_timestamps]
        
        # Obtaining the new values from interpolation
        before = [original_timestamps[0], original_values[0][0]]
        after = [original_timestamps[1], original_values[1][0]]
        new_values = [0.] * N
        j=0
        k=0
        
        for i in range(len(new_timestamps)):
            
            # Move forward in original table
            # Known point before interpolation point
            while (before[0] <= new_timestamps[i] and j<N-1):
                j+=1
                before[0] = original_timestamps[j]
            j-=1
            before[0] = original_timestamps[j]
            before[1] = original_values[j][0]
            # Known point after interpolation point
            while (after[0] <= new_timestamps[i] and k<N-1):
                k+=1
                after[0] = original_timestamps[k]
            after[1] = original_values[k][0]
                
            # Check the new date is sandwiched between the 2 original dates
            assert(before[0] <= new_timestamps[i])
            assert(new_timestamps[i] <= after[0])
            assert(j<=k)
            
            # Find the new value from interpolation
            slope = (after[1] - before[1]) / (after[0] - before[0])
            new_values[i] = before[1] + slope * (new_timestamps[i] - before[0])

        # Build the time series
        new_df = pd.DataFrame(index=new_index, data=new_values)
        new_ts = TimeSeries(new_df, tz=self.tz)
        
        return new_ts
        
    
    def decompose(self, polyn_order=None, start=None, end=None, 
                  extract_seasonality=False, period=None):
        """
        Performs a decomposition of the time series
        and returns the different components.
        
        Parameters
        ----------
        polyn_order : None or int
          Order of the polynomial when fitting a non-linear component.
        start_utc : str
          Starting date.
        end_utc : str
          Ending date.
        extract_seasonality : bool
          Option to extract seasonality signal.
        period : int
          Period of seasonality.
          
        Returns
        -------
        List of TimeSeries
          Content of the list depends on choices in arguments.
        """
        # Check
        if polyn_order is not None:
            try:
                assert(polyn_order>1)
            except AssertionError:
                raise AssertionError("polyn_order must be equal or more than 2.")
        
        # Prepare data in the specified period
        data = self.specify_data(start, end)
        X = [datetime.timestamp(x) for x in data.index]
        X = np.reshape(X, (len(X), 1))
        y = [data.values.tolist()[x][0] for x in range(len(data))]
        
        # Fit the linear component
        model = LinearRegression()
        model.fit(X, y)
        
        # Extract the linear trend
        lin_trend_y = model.predict(X)
        lin_trend_df = pd.DataFrame(index=data.index, data=lin_trend_y)
        lin_trend_ts = TimeSeries(lin_trend_df, tz=self.tz)
        
        # Remove the linear trend to the initial time series
        nonlin_y = y - lin_trend_y
        
        # Remove a polynomial component of a certain order
        if polyn_order is not None:
            polyn_model = make_pipeline(PolynomialFeatures(polyn_order), Ridge())
            polyn_model.fit(X, nonlin_y)
            polyn_component_y = polyn_model.predict(X)
            polyn_comp_df = pd.DataFrame(index=data.index, data=polyn_component_y)
            polyn_comp_ts = TimeSeries(polyn_comp_df, tz=self.tz)
        
        # Generate the resting part time series
        if polyn_order is not None:
            rest_y = nonlin_y - polyn_component_y
        else:
            rest_y = nonlin_y
        rest_df = pd.DataFrame(index=data.index, data=rest_y)
        rest_ts = TimeSeries(rest_df, tz=self.tz)
        
        # Extracting seasonality
        if extract_seasonality==True:
            # Receiving the period of seasonality in the residue
            try:
                assert(period)
                assert(isinstance(period, int))
            except AssertionError:
                raise AssertionError("Period must be specified for \
                                        extrac_seasonality=True mode.")
            P = period

            # Cut the series into seasonality-period chunks
            t = []
            if int(len(rest_y))%P==0:
                nchunks = int(len(rest_y))//P
            else:
                nchunks = int(len(rest_y))//P + 1

            for i in range(nchunks):
                if i == nchunks - 1:
                    t.append(rest_y[i*P:])
                else:
                    t.append(rest_y[i*P:i*P+P])

            # Do the average of the chunks
            t_avg = []
            for i in range(P):
                t_avg.append(np.mean([t[x][i] for x in range(nchunks)]))

            # Create a new series repeating this pattern
            seasonal_y = []
            for i in range(len(rest_y)):
                seasonal_y.append(t_avg[i%P])
            seasonal_df = pd.DataFrame(index=data.index, data=seasonal_y)
            seasonal_ts = TimeSeries(seasonal_df, tz=self.tz)

            # Build the residue time series
            residue_y = rest_y - seasonal_y
            residue_df = pd.DataFrame(index=data.index, data=residue_y)
            residue_ts = TimeSeries(residue_df, tz=self.tz)
        
        # Return results
        if polyn_order is not None:
            if extract_seasonality==True:
                return [lin_trend_ts, polyn_comp_ts, seasonal_ts, residue_ts]
            else:
                return [lin_trend_ts, polyn_comp_ts, rest_ts]
        else:
            if extract_seasonality==True:
                return [lin_trend_ts, seasonal_ts, residue_ts]
            else:
                return [lin_trend_ts, rest_ts]

    

    def gaussian_process(self, rbf_scale, rbf_scale_bounds, noise, noise_bounds,
                         alpha=1e-10, plotting=False, figsize=(12,5), dpi=100):
        """
        Employs Gaussian Process Regression (GPR) from scikit-learn to fit a time series. 
        
        Parameters
        rbf_scale : float
          Length scale for the RBF kernel.
        rbf_scale_bounds : 2-tuple of floats
          Length scale bounds for the RBF kernel.
        noise : float
          Noise level for the white noise kernel.
        noise_bounds : 2-tuple of floats
          Noise level bounds for the white noise kernel.
        alpha : float
          Noise added to the diagonal of the kernel matrix during fitting.
        plotting : bool
         Option to plot or not the result of the GPR.
        figsize : 2-tuple of ints
          Dimensions of the figure.
        dpi : int
          Dots-per-inch definition of the figure.
        
        Returns
        -------
        List of 3 TimeSeries
          3 time series for the mean and the envelope +sigma and -sigma of standard deviation.
        """

        # Shape the data
        X = np.array([float(datetime.timestamp(x)) for x in self.data.index])[:, np.newaxis]
        y = self.data.values.flatten()

        # Set the kernel
        initial_kernel = 1 * kernels.RBF(length_scale=rbf_scale,
                                         length_scale_bounds=rbf_scale_bounds) \
                         + kernels.WhiteKernel(noise_level=noise,
                                               noise_level_bounds=noise_bounds)

        # Do regression
        gpr = GaussianProcessRegressor(kernel=initial_kernel,
                                       alpha=alpha,
                                       optimizer='fmin_l_bfgs_b',
                                       n_restarts_optimizer=1,
                                       random_state=0)
        gpr = gpr.fit(X,y)
        print("The GPR score is: ", gpr.score(X,y))
        
        # Create fitting time series
        N = len(y)
        X_ = np.linspace(min(X)[0], max(X)[0], N)
        
        # Mean fit
        y_mean, y_cov = gpr.predict(X_[:,np.newaxis], return_cov=True)
        idx = self.data.index
        ts_mean = TimeSeries(pd.DataFrame(index=idx, data=y_mean), tz=self.tz, name='Mean from GPR')
        
        # Mean - (1-sigma)
        y_std_m = y_mean - np.sqrt(np.diag(y_cov))
        ts_std_m = TimeSeries(pd.DataFrame(index=idx, data=y_std_m), tz=self.tz, name='Mean-sigma from GPR')
        
        # Mean + (1-sigma)
        y_std_p = y_mean + np.sqrt(np.diag(y_cov))
        ts_std_p = TimeSeries(pd.DataFrame(index=idx, data=y_std_p), tz=self.tz, name='Mean+sigma from GPR')
        
        # Plot the result
        if plotting==True:
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(self.data.index, y_mean, color='k', lw=3)
            plt.plot(self.data.index, y_std_m, color='k')
            plt.plot(self.data.index, y_std_p, color='k')
            plt.fill_between(self.data.index, y_std_m, y_std_p, alpha=0.5, color='gray')
            plt.plot(self.data.index, self.data.values, color='r')
            title = "Gaussian Process Regression: \n Time series " \
                    + " from " + str(self.start_utc)[:10] + " to " + str(self.end_utc)[:10]
            plt.gca().set(title=title, xlabel="Date", ylabel="Value")
            plt.show()
        
        # Returning the time series
        return [ts_mean, ts_std_m, ts_std_p]


#---------#---------#---------#---------#---------#---------#---------#---------#---------#


# CLASS CatTimeSeries

class CatTimeSeries(Series):
    """
    Class defining a categoric time series and its methods.
    
    This class inherits from the parent class 'Series'.
    
    Attributes
    ----------
    data : DataFrame
      Contains a time-like index and for each time a single value.
    start_utc : Pandas.Timestamp
      Starting date.
    end_utc : Pandas.Timestamp
      Ending date.
    nvalues : int
      Number of values, i.e. also of dates.
    freq : str or None
      Frequency inferred from index.
    name : str
      Name or nickname of the series.
    unit : str or None
      Unit of the time series values.
    tz : str
      Timezone name.
    timezone : pytz timezone
      Timezone associated with dates.
    type : str
      Type of the series.
    """
    
    def __init__(self, df=None, tz=None, unit=None, name=""):
        """
        Receives a data frame as an argument and initializes the time series.
        """

        super().__init__(df=df, tz=tz, unit=unit, name=name)
        
        # Add attributes initialization if needed
        self.type = 'CatTimeSeries'

    
    def prepare_cat_plot(self):
        """
        Returns an appropriate dictionary to plot values of a CatTimeSeries.
        """
        
        # Initialization
        set_cats = sorted(list(set(self.data.values.flatten())))
        n_cats = len(set_cats)
        
        try:
            assert(n_cats<=10)
        except ValueError:
            raise ValueError("Number of categories too large for colors handling.")
        
        # Dates
        X = [datetime.timestamp(x) for x in self.data.index]
        # Adding one more step to show last value
        delta = self.data.index[-1] - self.data.index[-2]
        X.append(datetime.timestamp(self.data.index[-1] + delta))
        
        # Category values
        y = self.data.values.flatten().tolist()
        # Copying the last values
        y.append(y[-1])
            
        # Prepare Colors
        large_color_dict = { 0: 'Red', 1: 'DeepPink', 2: 'DarkOrange', 3: 'Yellow',
                             4: 'Magenta', 5: 'Lime', 6: 'Dark Green', 7: 'DarkCyan',
                             8: 'DarkTurquoise', 9:'DodgerBlue' }
        restricted_keys = [int(x) for x in np.linspace(0,9,n_cats).tolist()]
        restricted_colors = [large_color_dict[x] for x in restricted_keys]
        keys_to_cats = [set_cats[x] for x in range(0,n_cats)]

        # Create the restricted color dictionary
        D = dict(zip(keys_to_cats, restricted_colors))
        
        return X, y, D
    
    
    def simple_plot(self, figsize=(12,5), dpi=100):
        """
        Plots the categorical time series in a simple way.
        The number of categories is limited to 10 in order to easily handle colors.

        Parameters
        ----------
        figsize : 2-tuple of ints
          Dimensions of the figure.
        dpi : int
          Dots-per-inch definition of the figure.
          
        Returns
        -------
        None
          None
        """
        
        # Making the restricted color dictionary
        X, y, D = self.prepare_cat_plot()
        
        # Initiate figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Color block
        left_X = X[0]
        current_y = y[0]
        for i in range(1,self.nvalues+1,1):
            
            # For any block
            if y[i] != current_y:
                ax.fill_between([datetime.fromtimestamp(left_X), datetime.fromtimestamp(X[i])],
                                [0,0], [1,1], color=D[current_y], alpha=0.5)
                left_X = X[i]
                current_y = y[i]

            # For the last block
            if i == self.nvalues:
                ax.fill_between([datetime.fromtimestamp(left_X), datetime.fromtimestamp(X[i])],
                                [0,0], [1,1], color=D[current_y], alpha=0.5)
        
        # Make it cute
        title = "Categorical Time series " + self.name + " from " + str(self.start_utc)[:10] \
                + " to " + str(self.end_utc)[:10]
        plt.gca().set(title=title, xlabel="Date", ylabel="")
        plt.show()
        
        return None
    
    
#---------#---------#---------#---------#---------#---------#---------#---------#---------#
    

### FUNCTIONS HELPING TO CREATE A TIMESERIES ###

def type_to_series(type):
    """
    Returns the class TimeSeries or CatTimeSeries
    depending on wheter it receives 'TS' or 'CTS' argument.
    """
    
    if type == 'TS':
        return TimeSeries
    elif type == 'CTS':
        return CatTimeSeries
    
    if type == None:
        return TimeSeries
    else:
        raise ValueError("Series type not recognized.")
        

def build_from_csv(tz=None, unit=None, name=None, type=None, **kwargs):
    """
    Returns a list of time series from the reading of a .csv file.
    This function uses the function pandas.read_csv().
    
    Arguments
    ---------
    tz : str or list of str.
      Timezone name or list of timezone names.
    unit : str or list of str
      Unit name or list of unit names.
    name : str or list of str
      Time series name or list of time series names.
    type : str or list of str
      Time series type or list of time series types.
    **kwargs
        Arbitrary keyword arguments for pandas.read_csv().
    
    Returns
    -------
    List of TimeSeries
      Time series built from the .csv file.
      
    Notes
    -----
      To learn more about pandas.read_csv(), please refer to:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    """
   
    # Import data into a DataFrame
    df = pd.read_csv(**kwargs)
    ncols = df.shape[1]
        
    # Return a time series
    if ncols == 1 :
        return type_to_series(type=type)(df, tz=tz, unit=unit, name=name)
    
    # or return a list of time series
    else:
        # Checks
        if tz is not None:
            assert(isinstance(tz, list))
            assert(len(tz)==ncols)
        else:
            tz = [None] * ncols
            
        if unit is not None:
            assert(isinstance(unit, list))
            assert(len(unit)==ncols)
        else:
            unit = [None] * ncols
            
        if name is not None:
            assert(isinstance(name, list))
            assert(len(name)==ncols)
        else:
            name = [None] * ncols
            
        if type is not None:
            assert(isinstance(type, list))
            assert(len(type)==ncols)
        else:
            type = ['TS'] * ncols
            
        # Fill up a list with time series
        ts_list = []
        for i in range(ncols):
            ts_list.append( type_to_series(type[i])(pd.DataFrame(df.iloc[:,i]), tz=tz[i],
                                                                                unit=unit[i],
                                                                                name=name[i]) )
        return ts_list


def build_from_excel(tz=None, unit=None, name=None, type=None, **kwargs):
    """
    Returns a list of time series from the reading of an excel file.
    This function uses the function pandas.read_excel().
    
    Arguments
    ---------
    tz : str or list of str.
      Timezone name or list of timezone names.
    unit : str or list of str
      Unit name or list of unit names.
    name : str or list of str
      Time series name or list of time series names.
    type : str or list of str
      Time series type or list of time series types.
    **kwargs
        Arbitrary keyword arguments for pandas.read_excel().
    
    Returns
    -------
    List of TimeSeries
      Time series built from the excel file.
      
    Notes
    -----
      To learn more about pandas.read_excel(), please refer to:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
    """
   
    # Import data into a DataFrame
    df = pd.read_excel(**kwargs)
    ncols = df.shape[1]
        
    # Return a time series
    if ncols == 1 :
        return type_to_series(type=type)(df, tz=tz, unit=unit, name=name)
    # or return a list of time series
    else:
        # Checks
        if tz is not None:
            assert(isinstance(tz, list))
            assert(len(tz)==ncols)
        else:
            tz = [None] * ncols
            
        if unit is not None:
            assert(isinstance(unit, list))
            assert(len(unit)==ncols)
        else:
            unit = [None] * ncols
            
        if name is not None:
            assert(isinstance(name, list))
            assert(len(name)==ncols)
        else:
            name = [None] * ncols
            
        if type is not None:
            assert(isinstance(type, list))
            assert(len(type)==ncols)
        else:
            type = ['TS'] * ncols
            
        # Fill up a list with time series
        ts_list = []
        for i in range(ncols):
            ts_list.append( type_to_series(type[i])(pd.DataFrame(df.iloc[:,i]), tz=tz[i],
                                                                                unit=unit[i],
                                                                                name=name[i]) )
        return ts_list


    
def build_from_dataframe(df, tz=None, unit=None, name=None, type=None):
    """
    Returns a list of time series from the reading of Pandas DataFrame.
    
    Arguments
    ---------
    tz : str or list of str.
      Timezone name or list of timezone names.
    unit : str or list of str
      Unit name or list of unit names.
    name : str or list of str
      Time series name or list of time series names.
    type : str or list of str
      Time series type or list of time series types.
    **kwargs
        Arbitrary keyword arguments for pandas.read_csv().
    
    Returns
    -------
    List of TimeSeries
      Time series built from the Pandas DataFrame.
    """
   
    # Import data into a DataFrame
    ncols = df.shape[1]
        
    # Return a time series
    if ncols == 1 :
        return type_to_series(type=type)(df, tz=tz, unit=unit, name=name)
    
    # or return a list of time series
    else:
        # Checks
        if tz is not None:
            assert(isinstance(tz, list))
            assert(len(tz)==ncols)
        else:
            tz = [None] * ncols
            
        if unit is not None:
            assert(isinstance(unit, list))
            assert(len(unit)==ncols)
        else:
            unit = [None] * ncols
            
        if name is not None:
            assert(isinstance(name, list))
            assert(len(name)==ncols)
        else:
            name = [None] * ncols
            
        if type is not None:
            assert(isinstance(type, list))
            assert(len(type)==ncols)
        else:
            type = ['TS'] * ncols
            
        # Fill up a list with time series
        ts_list = []
        for i in range(ncols):
            ts_list.append( type_to_series(type[i])(pd.DataFrame(df.iloc[:,i]), tz=tz[i],
                                                                                unit=unit[i],
                                                                                name=name[i]) )
        return ts_list


def build_from_list(list_values, tz=None, unit=None, name="", **kwargs):
    """
    Returns a time series or categorical time series from the reading of a list.
    
    Parameters
    ----------
    list_values : list of float or str
      List of values to generate either a TimeSeries or a CatTimeSeries.
    unit : str
      Unit of the time series values when generating a TimeSeries.
    name : str
      Name or nickname of the series.
    **kwargs
        Arbitrary keyword arguments for pandas.date_range().
    
    Returns
    -------
    TimeSeries
      Time series built from the list of values and dates.
      
    Notes
    -----
      For pandas.date_range please consult the following page:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html
    """
   
    # Generate index
    data_index = pd.date_range(**kwargs)
    T = len(data_index)
    
    # Making DataFrame
    df = pd.DataFrame(index=data_index, data=list_values)
    
    # Checks
    try:
        assert(len(list_values)==T)
    except IndexError:
        raise IndexError("Size of the index does not equate the length of list_values.")
        
    # If the first value is a string, make a CatTimeSeries
    if type(list_values[0]) == str:
        ts = CatTimeSeries(df, tz=tz, unit=unit, name=name)
    # If the first value isn't a string, make a TimeSeries
    else:
        ts = TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return ts


def build_from_lists(list_dates, list_values, tz=None, unit=None, name=""):
    """
    Returns a time series or categorical time series from the reading of lists.
    
    Parameters
    ----------
    list_dates : list of timedates or str
      List of values to generate either a TimeSeries or a CatTimeSeries.
    list_values : list of float or str
      List of values to generate either a TimeSeries or a CatTimeSeries.
    unit : str
      Unit of the time series values when generating a TimeSeries.
    name : str
      Name or nickname of the series.
    
    Returns
    -------
    TimeSeries
      Time series built from the lists of values and dates.
    """

    # Checks
    try:
        assert(len(list_dates)==len(list_values))
    except IndexError:
        raise IndexError("Lengths of list_dates and list_values should be equal.")
    
    # Making DataFrame
    df = pd.DataFrame(index=list_dates, data=list_values)
        
    # If the first value is a string, make a CatTimeSeries
    if type(list_values[0]) == str:
        ts = CatTimeSeries(df, tz=tz, unit=unit, name=name)
    # If the first value isn't a string, make a TimeSeries
    else:
        ts = TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return ts




    
    
### FUNCTIONS USING TIMESERIES AS ARGUMENTS ###
    

def multi_plot(Series, figsize=(12,5), dpi=100):
    """
    Plots multiple time series together.
    
    Parameters
    ----------
    Series : List of Series
      Series (i.e. TimeSeries or CatTimeSeries) to be plotted.
    figsize : 2-tuple of ints
      Dimensions of the figure.
    dpi : int
      Dots-per-inch definition of the figure.
    """

    # Initialization
    N = len(Series)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    min_date, max_date = Series[0].data.index[0], Series[0].data.index[-1]
    min_val, max_val = min(Series[0].data.values.flatten()), max(Series[0].data.values.flatten())

    # Determine min and max values
    for i in range(1,N):
        min_date = min(Series[i].data.index[0], min_date)
        max_date = max(Series[i].data.index[-1], max_date)
        if Series[i].type == 'TimeSeries':
            min_val = min(min(Series[i].data.values.flatten()), min_val)
            max_val = max(max(Series[i].data.values.flatten()), max_val)

    
    # Loop through the series
    for i in range(N):
            
        # If the series is a CatTimeSeries:
        if Series[i].type == 'CatTimeSeries':
            # Get values and adapted dictionary
            X, y, D = Series[i].prepare_cat_plot()
            # Color block
            left_X = X[0]
            current_y = y[0]
            for i in range(1,len(X),1):
                # For any block
                if y[i] != current_y:
                    ax.fill_between([ datetime.fromtimestamp(left_X),
                                      datetime.fromtimestamp(X[i])],
                                    [min_val, min_val],
                                    [max_val, max_val],
                                    color=D[current_y],
                                    alpha=0.5)
                    left_X = X[i]
                    current_y = y[i]
                # For the last block
                if i == len(X)-1:
                    ax.fill_between([ datetime.fromtimestamp(left_X),
                                      datetime.fromtimestamp(X[i])],
                                    [min_val, min_val],
                                    [max_val, max_val],
                                    color=D[current_y],
                                    alpha=0.5)
            
        # If the series is a TimeSeries
        elif Series[i].type == 'TimeSeries':
            plt.plot(Series[i].data.index, Series[i].data.values)
        
    # Make it cute
    title = "Multiplot of time series from " + str(min_date)[:10] \
            + " to " + str(max_date)[:10]
    plt.gca().set(title=title, xlabel="Date", ylabel="Value")
    plt.show()
        
    return None


def multi_plot_distrib(Series, bins=20, figsize=(10,4), dpi=100):
    """
    Plots multiple time series together and their distributions of values.
    Only TimeSeries are allowed here, not CatTimeSeries.
    
    Parameters
    ----------
    Series : List of Series
      Series TimeSeries to be plotted.
    bins : int
      Number of bins for the distribution of values.
    figsize : 2-tuple of ints
      Dimensions of the figure.
    dpi : int
      Dots-per-inch definition of the figure.
      
    Returns
    -------
    None
      None
    """

    # Checks
    assert(isinstance(bins,int))

    # Initialization
    N = len(Series)
    min_date, max_date = Series[0].data.index[0], Series[0].data.index[-1]
    min_val, max_val = min(Series[0].data.values.flatten()), max(Series[0].data.values.flatten())

    # Determine min and max values
    for i in range(1,N):
        min_date = min(Series[i].data.index[0], min_date)
        max_date = max(Series[i].data.index[-1], max_date)
        if Series[i].type == 'TimeSeries':
            min_val = min(min(Series[i].data.values.flatten()), min_val)
            max_val = max(max(Series[i].data.values.flatten()), max_val)
        
    # Plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 4)

    # Plot 1 - Time Series simple plot
    f_ax1 = fig.add_subplot(gs[:, 0:3])
    # Loop through the series
    for i in range(N):
        f_ax1.plot(Series[i].data.index, Series[i].data.values)
    title1 = "Time series from " + str(min_date)[:10] + " to " + str(max_date)[:10]
    plt.gca().set(title=title1, xlabel="Date", ylabel="Value")

    # Plot 2 - Distribution of values
    f_ax2 = fig.add_subplot(gs[:, 3:])
    # Loop through the series
    for i in range(N):
        Series[i].data.hist(bins=bins, grid=False, ax=f_ax2, orientation="horizontal", alpha=N/(2*N-1))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0)
    title2 = "Distributions"
    plt.gca().set(title=title2, xlabel="Value", ylabel="Hits")

    return None
    

    
    
#---------#---------#---------#---------#---------#---------#---------#---------#---------#