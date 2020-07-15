# Created on 2020/7/15

# Packages
import numpy as np
import matplotlib.pyplot as plt


# CLASS timeseries
class timeseries:
    """
    Class defining a time series and its methods.
    """
    
    def __init__(self, df):
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
    
    
    def simple_plot(self, figsize=(12,5), dpi=100):
        """
        Simple function to plot the time series.

        Arguments:
        - figsize: size of the figure as tuple of 2 integers.
        - dpi: dots-per-inch definition of the figure.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.data.index, self.data.value, color='k')
        title = "Time series from " + str(self.start)[:10] + " to " + str(self.end)[:10]
        plt.gca().set(title=title, xlabel="Date", ylabel="Value")
        plt.show()
    
    
    def hist_avg(self, start=None, end=None):
        """
        Method that returns the historical average of the time series between two dates.
        Default is for the whole series.
        """
        if start==None and end==None:
            avg = self.data.values.mean()
        elif start==None and end!=None:
            avg = self.data[:end].values.mean()
        elif start!=None and end==None:
            avg = self.data[start:].values.mean()
        elif start!=None and end!=None:
            avg = self.data[start:end].values.mean()
        return avg
    
    
    def hist_std(self, start=None, end=None):
        """
        Method that returns the historical standard deviation of the time series between two dates.
        Default is for the whole series.
        """
        if start==None and end==None:
            std = self.data.values.std()
        elif start==None and end!=None:
            std = self.data[:end].values.std()
        elif start!=None and end==None:
            std = self.data[start:].values.std()
        elif start!=None and end!=None:
            std = self.data[start:end].values.std()
        return std
    
    
    def min(self, start=None, end=None):
        """
        Method that returns the minimum of the series.
        """
        if start==None and end==None:
            ts_min = self.data.values.min()
        elif start==None and end!=None:
            ts_min = self.data[:end].values.min()
        elif start!=None and end==None:
            ts_min = self.data[start:].values.min()
        elif start!=None and end!=None:
            ts_min = self.data[start:end].values.min()
        return ts_min
    
    
    def max(self, start=None, end=None):
        """
        Method that returns the maximum of the series.
        """
        if start==None and end==None:
            ts_max = self.data.values.max()
        elif start==None and end!=None:
            ts_max = self.data[:end].values.max()
        elif start!=None and end==None:
            ts_max = self.data[start:].values.max()
        elif start!=None and end!=None:
            ts_max = self.data[start:end].values.max()
        return ts_max
    
    
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
        if start==None and end==None:
            data = self.data
        elif start==None and end!=None:
            data = self.data[:end]
        elif start!=None and end==None:
            data = self.data[start:]
        elif start!=None and end!=None:
            data = self.data[start:end]
        
        # General case
        assert(l < data.shape[0])
        Numerator = np.mean((data - data.mean()) * (data.shift(l) - data.shift(l).mean()))
        Denominator = data.std() * data.shift(l).std()
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
        if start==None:
            s = str(self.start)[:10]
        else:
            s = start
        if end==None:
            e = str(self.end)[:10]
        else:
            s = end
        title = "Autocorrelation from " + s + " to " + e + " for lags = [" + str(lag_min) + "," + str(lag_max) + "]"
        plt.gca().set(title=title, xlabel="Lag", ylabel="Autocorrelation Value")
        plt.show()