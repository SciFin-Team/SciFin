# Standard library imports
import sys
new_path = sys.path[0][:-len("/examples/timeseries")]
sys.path.append(new_path)

# Third party imports
# /

# Local application imports
from scifin import timeseries as ts
from scifin import classifier as cl

# Build two time series of AR(1) type.
rs1 = ts.auto_regressive(start_date="2000-01-01", end_date="2020-01-01", frequency='M', start_values=[1.],
                         cst=1., order=1, coeffs=[0.3], sigma=0.1)

rs2 = ts.auto_regressive(start_date="2000-01-01", end_date="2020-01-01", frequency='M', start_values=[1.],
                         cst=1., order=1, coeffs=[0.35], sigma=0.1)

# Plot
ts.multi_plot([rs1, rs2])

# Display distance information
print()
print("COMPUTING DISTANCES:")
print("====================")
print(f"Euclidean distance: \t\t\t\t\t {cl.euclidean_distance(rs1, rs2)}")
print(f"Dynamic Time Warping distance with mode 'abs': \t\t {cl.dtw_distance(rs1, rs2, mode='abs')}")
print(f"Dynamic Time Warping distance with mode 'square': \t {cl.dtw_distance(rs1, rs2, mode='square')}")
print()

