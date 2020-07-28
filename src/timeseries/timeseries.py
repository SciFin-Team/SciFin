# Created on 2020/7/15

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats

from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import timeseries.timeseries as ts


# CLASS timeseries
class timeseries:
    """
    Class defining a time series and its methods.
    """
    
    def __init__(self, df, name=""):
        """
        Initializer. Function that receives a data frame as an argument and initialize the time series class.
        """
        # Making sure it is a single column data frame
        assert(df.shape[1]==1)
        # Extract values
        self.data = df
        self.start = df.index[0]
        self.end = df.index[-1]
        self.nvalues = df.shape[0]
        self.name = name
    
    
    def __specify_data(self, start, end):
        """
        Private method that returns the appropriate data according to user's specifying or not the start and end dates.
        """
        # Preparing data frame
        if start==None and end==None:
            data = self.data
        elif start==None and end!=None:
            data = self.data[:end]
        elif start!=None and end==None:
            data = self.data[start:]
        elif start!=None and end!=None:
            data = self.data[start:end]
        return data
    
    
    def __start_end_names(self, start, end):
        if start==None:
            s = str(self.start)[:10]
        else:
            s = start
        if end==None:
            e = str(self.end)[:10]
        else:
            e = end
        return s,e
    
    
    
    ### PLOTTING INFORMATION ABOUT THE TIME SERIES ###
    
    def simple_plot(self, figsize=(12,5), dpi=100):
        """
        Simple method to plot the time series.

        Arguments:
        - figsize: size of the figure as tuple of 2 integers.
        - dpi: dots-per-inch definition of the figure.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.data.index, self.data.values, color='k')
        title = "Time series " + self.name + " from " + str(self.start)[:10] + " to " + str(self.end)[:10]
        plt.gca().set(title=title, xlabel="Date", ylabel="Value")
        plt.show()
    
    
    def distribution(self, start=None, end=None, bins=20, figsize=(8,4), dpi=100):
        """
        Method that plots the distribution of values between two dates.
        """
        
        # Preparing data frame
        data = self.__specify_data(start, end)
            
        # Plotting its distribution of values
        plt.figure(figsize=figsize, dpi=dpi)
        data.hist(bins=bins, color='k', grid=False)
        s,e = self.__start_end_names(start, end)
        title = "Distribution of values between " + s + " and " + e
        plt.gca().set(title=title, xlabel="Value", ylabel="Hits")
        plt.show()
        
        
    def is_sampling_uniform(self):
        """
        Function that tests if the sampling of a time series is uniform or not.

        Returns a boolean value True, when the sampling is uniform, False otherwise.
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
                
    
    def get_sampling_interval(self):
        """
        Function that returns the sampling interval for a uniformly-sampled time series.
        """
        if(self.is_sampling_uniform()==False):
            print("Error: the timeseries is not uniformly sampled.")
        else:
            interval = datetime.timestamp(self.data.index[1]) - datetime.timestamp(self.data.index[0])
            return interval
        

    def lag_plot(self, lag=1, figsize=(5,5), dpi=100, alpha=0.5):
        """
        Function that returns the scatter plot x_t v.s. x_{t-l}.
        """
        # Check
        try:
            assert(lag>0)
        except AssertionError:
            raise AssertionError("The lag must be an integer equal or more than 1.")
        
        # Doing the plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        lag_plot(self.data, lag=lag, c='black', alpha=alpha)
        
        # Setting title
        title = "Lag plot of time series " + self.name
        plt.gca().set(title=title, xlabel="x(t)", ylabel="x(t+"+str(lag)+")")
        plt.show()
        
        
    def lag_plots(self, nlags=5, figsize=(10,10), dpi=100, alpha=0.5):
        """
        Function that returns a number of scatter plots x_t v.s. x_{t-l} where l is the lag value taken from [0,...,nlags].
        We require nlags > 1.
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
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize, dpi=dpi)
        for i, ax in enumerate(axes.flatten()[:nlags]):
            lag_plot(self.data, lag=i+1, ax=ax, c='black', alpha=alpha)
            ax.set_xlabel("x(t)")
            ax.set_ylabel("x(t+"+str(i+1)+")")
        
        # Setting title
        title = "Multiple lag plots of time series " + self.name
        fig.suptitle(title, )
        plt.show()
        
        
        
    
    
    ### SIMPLE DATA EXTRACTION ON THE TIME SERIES ###
    
    def hist_avg(self, start=None, end=None):
        """
        Method that returns the historical average of the time series between two dates (default is the whole series).
        """
        data = self.__specify_data(start, end)
        avg = data.values.mean()
        return avg
    
    
    def hist_std(self, start=None, end=None):
        """
        Method that returns the historical standard deviation of the time series between two dates (default is the whole series).
        """
        data = self.__specify_data(start, end)
        std = data.values.std()
        return std
    
    
    def hist_skew(self, start=None, end=None):
        """
        Method that returns the historical skew of the time series between two dates (default is the whole series).
        """
        data = self.__specify_data(start, end)
        skew = stats.skew(data.values)[0]
        return skew
    
    
    def hist_kurtosis(self, start=None, end=None):
        """
        Method that returns the historical (Fisher) kurtosis of the time series between two dates (default is the whole series).
        """
        data = self.__specify_data(start, end)
        kurt = stats.kurtosis(data.values)[0]
        return kurt
    
    
    def min(self, start=None, end=None):
        """
        Method that returns the minimum of the series.
        """
        data = self.__specify_data(start, end)
        ts_min = data.values.min()
        return ts_min
    
    
    def max(self, start=None, end=None):
        """
        Method that returns the maximum of the series.
        """
        data = self.__specify_data(start, end)
        ts_max = data.values.max()
        return ts_max
    
    
    def percent_change(self, start=None, end=None):
        """
        Method that returns the percent change of the series.
        """
        # Preparing data frame
        data = self.__specify_data(start, end)
        new_data = data.pct_change()
        new_ts = timeseries(new_data)
        return new_ts
        
    
    
    ### AUTOCORRELATION COMPUTATION ###
    
    def autocorrelation(self, lag=1, start=None, end=None):
        """
        Method returning the autocorrelation of the time series for a specified lag.
        We use the function $\rho_l = \frac{Cov(x_t, x_{t-l})}{\sqrt(Var[x_t] Var[x_{t-l}])} where $x_t$ is the time series at time t. Cov denotes the covariance and Var the variance. We also use the properties $\rho_0 = 1$ and $\rho_{-l} = \rho_l$ (choosing LaTeX notations here).
        """
        l = abs(lag)
        # Trivial case
        if l==0:
            return 1
        
        # Preparing data frame
        data = self.__specify_data(start, end)
        
        # General case
        assert(l < data.shape[0])
        shifted_data = data.shift(l)
        Numerator = np.mean((data - data.mean()) * (shifted_data - shifted_data.mean()))
        Denominator = data.std() * shifted_data.std()
        return Numerator / Denominator
    
    
    def plot_autocorrelation(self, lag_min=0, lag_max=25, start=None, end=None, figsize=(8,4), dpi=100):
        """
        Method making use of the autocorrelation method in order to return a plot of the autocorrelation againts the lag values.
        """
        
        assert(lag_max>lag_min)
        x_range = list(range(lag_min, lag_max+1, 1))
        ac = [self.autocorrelation(lag=x, start=start, end=end) for x in x_range]
        
        # Plotting
        plt.figure(figsize=figsize, dpi=dpi)
        plt.bar(x_range, ac, color='k')
        s,e = self.__start_end_names(start, end)
        title = "Autocorrelation from " + s + " to " + e + " for lags = [" + str(lag_min) + "," + str(lag_max) + "]"
        plt.gca().set(title=title, xlabel="Lag", ylabel="Autocorrelation Value")
        plt.show()
        
    
    def ACF_PACF(self, lag_max=25, figsize=(12,3), dpi=100):
        """
        Returns a plot of the AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF) from statsmodels.
        """
        # Plotting
        fig, axes = plt.subplots(1,2, figsize=figsize, dpi=dpi)
        plot_acf(self.data.values.tolist(), lags=lag_max, ax=axes[0])
        plot_pacf(self.data.values.tolist(), lags=lag_max, ax=axes[1])
        plt.show()
    
    
    
    ### SIMPLE TRANSFORMATIONS OF THE TIME SERIES TO CREATE A NEW TIME SERIES ###
    
    def trim(self, new_start, new_end):
        """
        Method that trims the time series to the desired dates and send back a new time series.
        """
        new_df = self.data[new_start:new_end]
        new_ts = timeseries(new_df)
        return new_ts
    
    
    def add_cst(self, cst=0):
        """
        Method that adds a constant to the time series.
        """
        new_df = self.data + cst
        new_ts = timeseries(new_df)
        return new_ts
    
    
    def mult_by_cst(self, cst=1):
        """
        Method that multiplies the time series by a constant.
        """
        new_df = self.data * cst
        new_ts = timeseries(new_df)
        return new_ts
    
    
    def linear_combination(self, other_timeseries, factor1=1, factor2=1):
        """
        Method that adds a timeseries to the current one according to linear combination: factor1 * current_ts + factor2 * other_timeseries.
        """
        new_df = factor1 * self.data + factor2 * other_timeseries.data
        new_ts = timeseries(new_df)
        return new_ts
    
    
    
    ### FITTING METHODS ###
    
    def rolling_avg(self, pts=1):
        """
        Method that transforms the time series into a rolling window average time series.
        """
        new_values = [self.data[x-pts+1:x].mean() for x in range(pts-1, self.nvalues, 1)]
        new_df = pd.DataFrame(index=self.data.index[pts-1:self.nvalues], data=new_values)
        new_ts = timeseries(new_df)
        return new_ts
    
    
    def polyfit(self, order=1, start=None, end=None):
        """
        Method that provides a polynomial fit of the time series.
        """
        data = self.__specify_data(start, end)
        new_index = [datetime.timestamp(x) for x in data.index]
        new_values = [data.values.tolist()[x][0] for x in range(len(data))]
        fit_formula = np.polyfit(new_index, new_values, deg=order)
        model = np.poly1d(fit_formula)
        print("Evaluated model: \n", model)
        yfit = [model(x) for x in new_index]
        assert(len(data.index)==len(yfit))
        new_df = pd.DataFrame(index=data.index, data=yfit)
        new_ts = timeseries(new_df)
        return new_ts

    
    def sample_uniformly(self):
        """
        Method that returns a new time series for which the sampling is uniform.
        """
        # Check actually we need to do something
        if self.is_sampling_uniform() == True:
            print("Time series already has a uniform sampling. Returning the same time series.")
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
        new_ts = timeseries(new_df)
        return new_ts
        
    
    
    def decompose(self, polyn_order=None, start=None, end=None, extract_seasonality=False, period=None):
        """
        Method that performs a decomposition of the time series and returns the different components.
        """
        # Check
        if polyn_order != None:
            try:
                assert(polyn_order>1)
            except AssertionError:
                raise AssertionError("polyn_order must be equal or more than 2.")
        
        # Preparing data in the specified period
        data = self.__specify_data(start, end)
        X = [datetime.timestamp(x) for x in data.index]
        X = np.reshape(X, (len(X), 1))
        y = [data.values.tolist()[x][0] for x in range(len(data))]
        
        # Fitting the linear component
        model = LinearRegression()
        model.fit(X, y)
        
        # Extract the linear trend
        lin_trend_y = model.predict(X)
        lin_trend_df = pd.DataFrame(index=data.index, data=lin_trend_y)
        lin_trend_ts = ts.timeseries(lin_trend_df)
        
        # Remove the linear trend to the initial time series
        nonlin_y = y - lin_trend_y
        
        # Remove a polynomial component of a certain order
        if polyn_order != None:
            polyn_model = make_pipeline(PolynomialFeatures(polyn_order), Ridge())
            polyn_model.fit(X, nonlin_y)
            polyn_component_y = polyn_model.predict(X)
            polyn_comp_df = pd.DataFrame(index=data.index, data=polyn_component_y)
            polyn_comp_ts = ts.timeseries(polyn_comp_df)
        
        # Generating the resting part time series
        if polyn_order != None:
            rest_y = nonlin_y - polyn_component_y
        else:
            rest_y = nonlin_y
        rest_df = pd.DataFrame(index=data.index, data=rest_y)
        rest_ts = ts.timeseries(rest_df)
        
        # Extracting seasonality
        if extract_seasonality==True:
            # Receiving the period of seasonality in the residue
            try:
                assert(period)
            except AssertionError:
                raise AssertionError("Period must be specified for extrac_seasonality=True mode.")
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
            seasonal_ts = ts.timeseries(seasonal_df)

            # Building the residue time series
            residue_y = rest_y - seasonal_y
            residue_df = pd.DataFrame(index=data.index, data=residue_y)
            residue_ts = ts.timeseries(residue_df)
        
        # Return results
        if polyn_order != None:
            if extract_seasonality==True:
                return [lin_trend_ts, polyn_comp_ts, seasonal_ts, residue_ts]
            else:
                return [lin_trend_ts, polyn_comp_ts, rest_ts]
        else:
            if extract_seasonality==True:
                return [lin_trend_ts, seasonal_ts, residue_ts]
            else:
                return [lin_trend_ts, rest_ts]

        
    
    


### FUNCTIONS USING TIMESERIES AS ARGUMENTS ###
    
    
def multi_plot(timeseries, figsize=(12,5), dpi=100):
    """
    Function that plots multiple time series together.
    """
    
    plt.figure(figsize=figsize, dpi=dpi)
    min_date = timeseries[0].data.index[0]
    max_date = timeseries[0].data.index[-1]
    for i in range(len(timeseries)):
        if timeseries[i].data.index[0] < min_date:
            min_date = timeseries[i].data.index[0]
        if timeseries[i].data.index[-1] > max_date:
            max_date = timeseries[i].data.index[-1]
        plt.plot(timeseries[i].data.index, timeseries[i].data.values)
    title = "Multiplot of time series from " + str(min_date)[:10] + " to " + str(max_date)[:10]
    plt.gca().set(title=title, xlabel="Date", ylabel="Value")
    plt.show()
        
        
        
