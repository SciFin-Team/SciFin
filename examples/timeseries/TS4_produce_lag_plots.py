# Standard library imports
# /

# Third party imports
# /

# Local application imports
from scifin.timeseries.timeseries import build_from_csv

# Build a time series from a CSV file online
ts1 = build_from_csv(filepath_or_buffer='https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                        parse_dates=['date'], index_col='date', unit="Number of sales", name="Sales_TimeSeries")

# Produce a single lag plot with a lag of 10
ts1.lag_plot(lag=10)

# Produce a plot with several lag plots from 0 to 10
ts1.lag_plots(nlags=10)


