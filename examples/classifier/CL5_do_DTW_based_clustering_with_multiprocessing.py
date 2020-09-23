# Standard library imports
import time

# Third party imports
# /

# Local application imports
from scifin import timeseries as ts
from scifin import montecarlo as mc
from scifin import classifier as cl

# Generate a list of random walks
N = 20
mc1 = mc.generate_series(n=N, names_base="RW-", series_model=ts.random_walk,
                         start_date="2000-01-01", end_date="2020-01-01", frequency='M',
                         start_value=1., sigma=0.01)

# Compute DTW distance matrix without multi-processing
t1 = time.time()
obs1_noMP = cl.dtw_distance_matrix_from_ts(mc1, window=10, mode='square', normalize=True)
t2 = time.time()
print(f"No multi-processing, time to compute: {t2-t1}")

# Compute DTW distance matrix with multi-processing
t1 = time.time()
obs1_withMP = cl.dtw_distance_matrix_from_ts_multi_proc(mc1, window=10, mode='square', normalize=True, n_proc=2)
t2 = time.time()
print(f"With multi-processing, time to compute: {t2-t1}")

# Check of equality of results
if (obs1_noMP == obs1_withMP).all().all():
    print("The 2 DTW matrices are the same!")
else:
    raise AssertionError("The 2 DTW matrices are different!")

