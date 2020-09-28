# Standard library imports
# /

# Third party imports
# /

# Local application imports
from scifin import timeseries as ts

# Build a time series from a CSV file online
ts1 = ts.build_from_csv(filepath_or_buffer='https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                        parse_dates=['date'], index_col='date', unit="Number of sales", name="Sales_TimeSeries")

# Build a rolling average time series
ts2 = ts1.rolling_avg(pts=10)

# Plot both time series
ts.multi_plot([ts1, ts2])


