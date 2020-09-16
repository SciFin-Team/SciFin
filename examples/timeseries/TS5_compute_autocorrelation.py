# Standard library imports
import sys
new_path = sys.path[0][:-len("/examples/timeseries")]
sys.path.append(new_path)
import warnings
warnings.filterwarnings(("ignore"))

# Third party imports
# /

# Local application imports
from scifin import timeseries as ts

# Build a time series from a CSV file online
ts1 = ts.build_from_csv(filepath_or_buffer='https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                        parse_dates=['date'], index_col='date', unit="Number of sales", name="Sales_TimeSeries")

# Produce a plot with autocorrelation
ts1.plot_autocorrelation(lag_max=100)

# Compare it with statsmodels acf calculation
ts1.acf_pacf(lag_max=100)


