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

# Make a decomposition of ts1
ts1_components = ts1.decompose(polyn_order=2, extract_seasonality=True, period=12)

# Join ts1 with its decomposition components
together = [ts1]
for tsi in ts1_components:
    together.append(tsi)

# Plot them together
ts.multi_plot(together)



