# Standard library imports
import sys
new_path = sys.path[0][:-len("/examples/timeseries")]
sys.path.append(new_path)
import warnings
warnings.filterwarnings(("ignore"))

# Third party imports
import pandas as pd

# Local application imports
from scifin import timeseries as ts

# Build a time series from a CSV file online
ts1 = ts.build_from_csv(filepath_or_buffer='https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                        parse_dates=['date'], index_col='date', unit="Number of sales", name="Sales_TimeSeries")

# Define min and max values of a range
range_min = 10
range_max = 20

# Create a DataFrame with categories
cts2_idx = ts1.data.index
cts2_vals = []
for x in range(ts1.nvalues):
    if ts1.data.values[x] >= range_min and ts1.data.values[x] <= range_max:
        cts2_vals.append('In Range')
    else:
        cts2_vals.append('Out of Range')
cts2_df = pd.DataFrame(index=cts2_idx, data=cts2_vals)

# Build a CatTimeSeries from it
cts2 = ts.CatTimeSeries(cts2_df)

# Plot them together
ts.multi_plot([ts1, cts2])



