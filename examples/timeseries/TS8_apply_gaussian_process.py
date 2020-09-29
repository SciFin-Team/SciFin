# Standard library imports
# /

# Third party imports
# /

# Local application imports
from scifin.timeseries.timeseries import build_from_csv

# Build a time series from a CSV file online
ts1 = build_from_csv(filepath_or_buffer='https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                     parse_dates=['date'], index_col='date', unit="Number of sales", name="Sales_TimeSeries")

# Apply Gaussian Process
[ts1_gp_mean, ts1_gp_m, ts1_gp_p] = ts1.gaussian_process(rbf_scale=1e6,
                                                         rbf_scale_bounds=(1e-2, 1e9),
                                                         noise=1e-5,
                                                         noise_bounds=(1e-10, 1e+3),
                                                         plotting=True)

# Change the scale bounds
[ts1_gp_mean, ts1_gp_m, ts1_gp_p] = ts1.gaussian_process(rbf_scale=1e6,
                                                         rbf_scale_bounds=(1e-1, 1e6),
                                                         noise=1e-5,
                                                         noise_bounds=(1e-10, 1e+3),
                                                         plotting=True)