# Standard library imports
# /

# Third party imports
# /

# Local application imports
from scifin.timeseries.timeseries import build_from_csv

# Build a time series from a CSV file online
ts1 = build_from_csv(filepath_or_buffer='https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                        parse_dates=['date'], index_col='date', unit="Number of sales", name="Sales_TimeSeries")

# Plot the distribution
ts1.distribution(bins=20)

# Plot the density
ts1.density(bins=20)


