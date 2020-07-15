# Created on 2020/7/15

# Packages
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


