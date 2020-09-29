# Standard library imports
# Standard library imports
# /

# Third party imports
# /

# Local application imports
from scifin.timeseries.timeseries import multi_plot
from scifin.timeseries.randomseries import auto_regressive
from scifin.classifier.classifier import euclidean_distance, dtw_distance

# Build two time series of AR(1) type.
rs1 = auto_regressive(start_date="2000-01-01", end_date="2020-01-01", frequency='M', start_values=[1.],
                      cst=1., order=1, coeffs=[0.3], sigma=0.1)

rs2 = auto_regressive(start_date="2000-01-01", end_date="2020-01-01", frequency='M', start_values=[1.],
                      cst=1., order=1, coeffs=[0.35], sigma=0.1)

# Plot
multi_plot([rs1, rs2])

# Display distance information
print()
print("COMPUTING DISTANCES:")
print("====================")
print(f"Euclidean distance: \t\t {euclidean_distance(rs1, rs2)}")
print(f"Dynamic Time Warping distance with mode 'abs': \t\t {dtw_distance(rs1, rs2, mode='abs')}")
print(f"Dynamic Time Warping distance with mode 'square': \t\t {dtw_distance(rs1, rs2, mode='square')}")
print()

