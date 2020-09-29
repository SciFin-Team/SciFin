# Standard library imports
# /

# Third party imports
# /

# Local application imports
from scifin.timeseries.timeseries import build_from_csv, multi_plot

# Build a time series from a CSV file online
ts1 = build_from_csv(filepath_or_buffer='https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                     parse_dates=['date'], index_col='date', unit="Number of sales", name="Sales_TimeSeries")

# Make a decomposition of ts1
ts1_components = ts1.decompose(polyn_order=2, extract_seasonality=True, period=12)

# Join ts1 with its decomposition components
ts1_and_components = [ts1]
for tsi in ts1_components:
    ts1_and_components.append(tsi)

# Plot them together
multi_plot(ts1_and_components)



