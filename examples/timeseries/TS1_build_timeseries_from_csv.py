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
                        parse_dates=['date'], index_col='date', unit="Number of sales", name="Time Series from CSV")

# Display information
print()
print("INFORMATION ABOUT THE TIME SERIES:")
print("==================================")
print("Starting date: \t\t", ts1.start_utc)
print("Ending date: \t\t", ts1.end_utc)
print("Number of values: \t", ts1.nvalues)
print("Frequency of data: \t", ts1.freq)
print("Unit of data: \t\t", ts1.unit)
print("Name: \t\t\t", ts1.name)
print()

# Plot it
ts1.simple_plot()
