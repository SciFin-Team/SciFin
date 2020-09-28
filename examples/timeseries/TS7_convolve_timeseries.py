# Standard library imports
# /

# Third party imports
# /

# Local application imports
from scifin import timeseries as ts

# Build a time series from a CSV file online
ts1 = ts.build_from_csv(filepath_or_buffer='https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                        parse_dates=['date'], index_col='date', unit="Number of sales", name="Sales_TimeSeries")

# Define Gaussian function
def Gaussian(x):
    return np.exp(-x**2/2) / np.sqrt(2*np.pi)

# Convolve it with ts1
ts2 = ts1.convolve(func=Gaussian, x_min=-2, x_max=2, n_points=10, normalize=True,
                   name='ts1-Convolved-with-Gaussian')
ts.multi_plot([ts1, ts2])

# Define Square function
def Square(x):
    if x>-1 and x<1:
        return 0.5
    else:
        return 0

# Convolve it with ts1
ts3 = ts1.convolve(func=Square, x_min=-2, x_max=2, n_points=10, normalize=True,
                   name='ts1-Convolved-with-Square')
ts.multi_plot([ts1, ts3])


