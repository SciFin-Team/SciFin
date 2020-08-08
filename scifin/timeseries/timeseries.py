# Created on 2020/7/15

# This module is for the class TimeSeries and related functions.

# Standard library imports
from datetime import datetime

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


#---------#---------#---------#---------#---------#---------#---------#---------#---------#

# CLASS Series

class Series:
    """
    Abstract class defining a Series and its methods.
    
    This class serves as a parent class for TimeSeries and CatTimeSeries.
    
    Attributes
    ----------
    data : DataFrame
      Contains a time-like index and for each time a single value.
    start : Pandas.Timestamp
      Starting date.
    end : Pandas.Timestamp
      Ending date.
    nvalues : int
      Number of values, i.e. also of dates.
    name : str
      Name of nickname of the series.
    """
    
    def __init__(self, df=None, name=""):
        """
        Receives a data frame as an argument and initializes the time series.
        """
        
        if (df is None) or (df.empty == True):
            
            self.data = pd.DataFrame(index=None, data=None)
            self.start = None
            self.end = None
            self.nvalues = 0
            self.name = 'Empty TimeSeries'
        
        else:
            
            # Making sure the dataframe is just
            # an index + 1 value column
            assert(df.shape[1]==1)
            
            # Extract values
            self.data = df
            self.start = df.index[0]
            self.end = df.index[-1]
            self.nvalues = df.shape[0]
            self.name = name
    

    def specify_data(self, start, end):
        """
        Returns the appropriate data according to user's specifying
        or not the desired start and end dates.
        """
         
        # Preparing data frame
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
        s = str(self.start)[:10] if (start is None) else start
        e = str(self.end)[:10] if (end is None) else end

        return s,e
    
    
    def is_sampling_uniform(self):
        """
        Tests if the sampling of a time series is uniform or not.
        Returns a boolean value True when the sampling is uniform, False otherwise.
        """
        # Preparing data
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
    start : Pandas.Timestamp
      Starting date.
    end : Pandas.Timestamp
      Ending date.
    nvalues : int
      Number of values, i.e. also of dates.
    name : str
      Name of nickname of the series.
    type : str
      Type of the series.
    """
    
    def __init__(self, df=None, name=""):
        """
        Receives a data frame as an argument and initializes the time series.
        """

        super().__init__(df=df, name=name)
        
        # Add attributes initialization if needed
        self.type = 'TimeSeries'
    
    
    
    ### PLOTTING INFORMATION ABOUT THE TIME SERIES ###
    
    def simple_plot(self, figsize=(12,5), dpi=100):
        """
        Plots the time series in a simple way.

        Parameters
        ----------
        figsize : 2-tuple of ints
          Dimensions of the figure.
        dpi : int
          Dots-per-inch definition of the figure.
        """
        
        # Plotting
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.data.index, self.data.values, color='k')
        
        # Make it cute
        title = "Time series " + self.name + " from " + str(self.start)[:10] \
                + " to " + str(self.end)[:10]
        plt.gca().set(title=title, xlabel="Date", ylabel="Value")
        plt.show()
        
        return None
    
    
    def distribution(self, start=None, end=None, bins=20, figsize=(8,4), dpi=100):
        """
        Plots the distribution of values between two dates.
        """
        
        # Preparing data frame
        data = self.specify_data(start, end)
            
        # Plotting distribution of values
        plt.figure(figsize=figsize, dpi=dpi)
        data.hist(bins=bins, color='k', grid=False)
        
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
        
        # Preparing data frame
        data = self.specify_data(start, end)
        s,e = self.start_end_names(start, end)
        
        # Plotting distribution of values
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        data.plot.density(color='k', ax=ax, legend=False)
        
        # Make it cute
        title = "Density plot of values between " + s + " and " + e
        plt.gca().set(title=title, xlabel="Value", ylabel="Density")
        plt.show()
        
        return None
    

    def plot_series_distrib(self, start=None, end=None, bins=20, figsize=(10,4), dpi=100):
        """
        Plots the time series and its associated distribution of values between two dates.
        """

        # Preparing data frame
        data = self.specify_data(start, end)
        s,e = self.start_end_names(start, end)
        
        # Plotting
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(1, 4)
        
        # Plot 1 - Time Series simple plot
        f_ax1 = fig.add_subplot(gs[:, 0:3])
        f_ax1.plot(data.index, data.values, color='k')
        title1 = "Time series " + self.name + " from " + s + " to " + e
        plt.gca().set(title=title1, xlabel="Date", ylabel="Value")
        
        # Plot 2 - distribution of values
        f_ax2 = fig.add_subplot(gs[:, 3:])
        data.hist(bins=bins, color='k', grid=False, ax=f_ax2, orientation="horizontal")
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
        
        # Doing the plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        pd.plotting.lag_plot(self.data, lag=lag, c='black', alpha=alpha)
        
        # Setting title
        title = "Lag plot of time series " + self.name
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
        
        # Doing the plots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                                 figsize=figsize, dpi=dpi)
        for i, ax in enumerate(axes.flatten()[:nlags]):
            pd.plotting.lag_plot(self.data, lag=i+1, ax=ax, c='black', alpha=alpha)
            ax.set_xlabel("x(t)")
            ax.set_ylabel("x(t+"+str(i+1)+")")
        
        # Setting title
        title = "Multiple lag plots of time series " + self.name
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
    
    
    def hist_skew(self, start=None, end=None):
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
        kurt = stats.kurtosis(data.values)[0]
        
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
    
    
    def percent_change(self, start=None, end=None):
        """
        Returns the percent change of the series.
        """
        data = self.specify_data(start, end)
        new_data = data.pct_change()
        new_ts = TimeSeries(new_data)
        
        return new_ts
    
    
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
        
        # Preparing data frame
        data = self.specify_data(start, end)
        
        # General case
        assert(l < data.shape[0])
        shifted_data = data.shift(l)
        Numerator = np.mean((data - data.mean()) * (shifted_data - shifted_data.mean()))
        Denominator = data.std() * shifted_data.std()
        
        return Numerator / Denominator
    
    
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
        
        # Plotting
        plt.figure(figsize=figsize, dpi=dpi)
        plt.bar(x_range, ac, color='k')
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
        # Plotting
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
        new_ts = TimeSeries(new_df)
        
        return new_ts
    
    
    def add_cst(self, cst=0):
        """
        Method that adds a constant to the time series.
        """
        new_df = self.data + cst
        new_ts = TimeSeries(new_df)
        
        return new_ts
    
    
    def mult_by_cst(self, cst=1):
        """
        Method that multiplies the time series by a constant.
        """
        new_df = self.data * cst
        new_ts = TimeSeries(new_df)
        
        return new_ts
    
    
    def linear_combination(self, OtherTimeSeries, factor1=1, factor2=1):
        """
        Method that adds a time series to the current one
        according to linear combination:
        factor1 * current_ts + factor2 * OtherTimeSeries.
        """
        new_df = factor1 * self.data + factor2 * OtherTimeSeries.data
        new_ts = TimeSeries(new_df)
        
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
        
        # Generating convolved values
        convolved_vals = np.convolve(func_vals, ts.flatten(), mode='same')
        convolved_ts = TimeSeries(pd.DataFrame(index=self.data.index, data=convolved_vals))
        
        return convolved_ts
    
    
    ### FITTING METHODS ###
    
    def rolling_avg(self, pts=1):
        """
        Transforms the time series into a rolling window average time series.
        """
        new_values = [self.data[x-pts+1:x].mean() for x in range(pts-1, self.nvalues, 1)]
        new_df = pd.DataFrame(index=self.data.index[pts-1:self.nvalues], data=new_values)
        new_ts = TimeSeries(new_df)
        
        return new_ts
    
    
    def polyfit(self, order=1, start=None, end=None):
        """
        Provides a polynomial fit of the time series.
        """
        
        # Preparing data
        data = self.specify_data(start, end)
        new_index = [datetime.timestamp(x) for x in data.index]
        new_values = [data.values.tolist()[x][0] for x in range(len(data))]
        
        # Doing the fit
        fit_formula = np.polyfit(new_index, new_values, deg=order)
        model = np.poly1d(fit_formula)
        print("Evaluated model: \n", model)
        yfit = [model(x) for x in new_index]
        
        # Building data frame
        assert(len(data.index)==len(yfit))
        new_df = pd.DataFrame(index=data.index, data=yfit)
        new_ts = TimeSeries(new_df)
        
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
        
        # Preparing the new index
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

        # Building the time series
        new_df = pd.DataFrame(index=new_index, data=new_values)
        new_ts = TimeSeries(new_df)
        
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
        start : str
          Starting date.
        end : str
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
        
        # Preparing data in the specified period
        data = self.specify_data(start, end)
        X = [datetime.timestamp(x) for x in data.index]
        X = np.reshape(X, (len(X), 1))
        y = [data.values.tolist()[x][0] for x in range(len(data))]
        
        # Fitting the linear component
        model = LinearRegression()
        model.fit(X, y)
        
        # Extract the linear trend
        lin_trend_y = model.predict(X)
        lin_trend_df = pd.DataFrame(index=data.index, data=lin_trend_y)
        lin_trend_ts = TimeSeries(lin_trend_df)
        
        # Remove the linear trend to the initial time series
        nonlin_y = y - lin_trend_y
        
        # Remove a polynomial component of a certain order
        if polyn_order is not None:
            polyn_model = make_pipeline(PolynomialFeatures(polyn_order), Ridge())
            polyn_model.fit(X, nonlin_y)
            polyn_component_y = polyn_model.predict(X)
            polyn_comp_df = pd.DataFrame(index=data.index, data=polyn_component_y)
            polyn_comp_ts = TimeSeries(polyn_comp_df)
        
        # Generating the resting part time series
        if polyn_order is not None:
            rest_y = nonlin_y - polyn_component_y
        else:
            rest_y = nonlin_y
        rest_df = pd.DataFrame(index=data.index, data=rest_y)
        rest_ts = TimeSeries(rest_df)
        
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

            # Cutting the series into seasonality-period chunks
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

            # Doing the average of the chunks
            t_avg = []
            for i in range(P):
                t_avg.append(np.mean([t[x][i] for x in range(nchunks)]))

            # Creating a new series repeating this pattern
            seasonal_y = []
            for i in range(len(rest_y)):
                seasonal_y.append(t_avg[i%P])
            seasonal_df = pd.DataFrame(index=data.index, data=seasonal_y)
            seasonal_ts = TimeSeries(seasonal_df)

            # Building the residue time series
            residue_y = rest_y - seasonal_y
            residue_df = pd.DataFrame(index=data.index, data=residue_y)
            residue_ts = TimeSeries(residue_df)
        
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

        # Shaping the data
        X = np.array([float(datetime.timestamp(x)) for x in self.data.index])[:, np.newaxis]
        y = self.data.values.flatten()

        # Setting the kernel
        initial_kernel = 1 * kernels.RBF(length_scale=rbf_scale,
                                         length_scale_bounds=rbf_scale_bounds) \
                         + kernels.WhiteKernel(noise_level=noise,
                                               noise_level_bounds=noise_bounds)

        # Doing regression
        gpr = GaussianProcessRegressor(kernel=initial_kernel,
                                       alpha=alpha,
                                       optimizer='fmin_l_bfgs_b',
                                       n_restarts_optimizer=1,
                                       random_state=0)
        gpr = gpr.fit(X,y)
        print("The GPR score is: ", gpr.score(X,y))
        
        # Creating fitting time series
        N = len(y)
        X_ = np.linspace(min(X)[0], max(X)[0], N)
        
        # Mean fit
        y_mean, y_cov = gpr.predict(X_[:,np.newaxis], return_cov=True)
        idx = self.data.index
        ts_mean = TimeSeries(pd.DataFrame(index=idx, data=y_mean), name='Mean from GPR')
        
        # Mean - (1-sigma)
        y_std_m = y_mean - np.sqrt(np.diag(y_cov))
        ts_std_m = TimeSeries(pd.DataFrame(index=idx, data=y_std_m), name='Mean-sigma from GPR')
        
        # Mean + (1-sigma)
        y_std_p = y_mean + np.sqrt(np.diag(y_cov))
        ts_std_p = TimeSeries(pd.DataFrame(index=idx, data=y_std_p), name='Mean+sigma from GPR')
        
        # Plotting the result
        if plotting==True:
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(self.data.index, y_mean, color='k', lw=3)
            plt.plot(self.data.index, y_std_m, color='k')
            plt.plot(self.data.index, y_std_p, color='k')
            plt.fill_between(self.data.index, y_std_m, y_std_p, alpha=0.5, color='gray')
            plt.plot(self.data.index, self.data.values, color='r')
            title = "Gaussian Process Regression: \n Time series " \
                    + " from " + str(self.start)[:10] + " to " + str(self.end)[:10]
            plt.gca().set(title=title, xlabel="Date", ylabel="Value")
            plt.show()
        
        # Returning the time series
        return [ts_mean, ts_std_m, ts_std_p]



    
    
#---------#---------#---------#---------#---------#---------#---------#---------#---------#
    
    
### FUNCTIONS USING TIMESERIES AS ARGUMENTS ###
    
    
def multi_plot(Series, figsize=(12,5), dpi=100):
    """
    Function that plots multiple time series together.
    
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
                    ax.fill_between([datetime.fromtimestamp(left_X), datetime.fromtimestamp(X[i])],
                                    [min_val, min_val], [max_val, max_val], color=D[current_y], alpha=0.5)
                    left_X = X[i]
                    current_y = y[i]
                # For the last block
                if i == len(X)-1:
                    ax.fill_between([datetime.fromtimestamp(left_X), datetime.fromtimestamp(X[i])],
                                    [min_val, min_val], [max_val, max_val], color=D[current_y], alpha=0.5)
            
        # If the series is a TimeSeries
        elif Series[i].type == 'TimeSeries':
            plt.plot(Series[i].data.index, Series[i].data.values)
        
    # Make it cute
    title = "Multiplot of time series from " + str(min_date)[:10] \
            + " to " + str(max_date)[:10]
    plt.gca().set(title=title, xlabel="Date", ylabel="Value")
    plt.show()
        
    return None



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
    start : Pandas.Timestamp
      Starting date.
    end : Pandas.Timestamp
      Ending date.
    nvalues : int
      Number of values, i.e. also of dates.
    name : str
      Name of nickname of the series.
    type : str
      Type of the series.
    """
    
    def __init__(self, df=None, name=""):
        """
        Receives a data frame as an argument and initializes the time series.
        """

        super().__init__(df=df, name=name)
        
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
        
        X = [datetime.timestamp(x) for x in self.data.index]
        y = self.data.values.flatten()
            
        # Preparing Colors
        large_color_dict = { 0: 'Red', 1: 'DeepPink', 2: 'DarkOrange', 3: 'Yellow',
                             4: 'Magenta', 5: 'Lime', 6: 'Dark Green', 7: 'DarkCyan',
                             8: 'DarkTurquoise', 9:'DodgerBlue' }
        restricted_keys = [int(x) for x in np.linspace(0,9,n_cats).tolist()]
        restricted_colors = [large_color_dict[x] for x in restricted_keys]
        keys_to_cats = [set_cats[x] for x in range(0,n_cats)]

        # Creating the restricted color dictionary
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
        """
        
        # Making the restricted color dictionary
        X, y, D = self.prepare_cat_plot()
        
        # Initiate figure
        #plt.figure(figsize=figsize, dpi=dpi)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Color block
        left_X = X[0]
        current_y = y[0]
        for i in range(1,self.nvalues,1):
            
            # For any block
            if y[i] != current_y:
                ax.fill_between([datetime.fromtimestamp(left_X), datetime.fromtimestamp(X[i])],
                                [0,0], [1,1], color=D[current_y], alpha=0.5)
                left_X = X[i]
                current_y = y[i]

            # For the last block
            if i == self.nvalues-1:
                ax.fill_between([datetime.fromtimestamp(left_X), datetime.fromtimestamp(X[i])],
                                [0,0], [1,1], color=D[current_y], alpha=0.5)
        
        # Make it cute
        title = "Categorical Time series " + self.name + " from " + str(self.start)[:10] \
                + " to " + str(self.end)[:10]
        plt.gca().set(title=title, xlabel="Date", ylabel="")
        plt.show()
        
        return None
    
    
    
#---------#---------#---------#---------#---------#---------#---------#---------#---------#