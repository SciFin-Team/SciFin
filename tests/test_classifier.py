# Import third party packages
import numpy as np

# Import my package
from scifin import timeseries as ts
from scifin.classifier import classifier as cl


class TestDistances:
    """
    Tests the functions euclidean_distance() and dtw_distance().
    """

    def test_distances_value(self):
        # Build two constant time series
        rs1 = ts.constant(start_date="2000-01-01", end_date="2020-01-01", frequency='Y', cst=1.)
        rs2 = ts.constant(start_date="2000-01-01", end_date="2020-01-01", frequency='Y', cst=2.)
        
        # Test distances are the same
        assert cl.euclidean_distance(rs1, rs2) == np.sqrt(rs1.nvalues)
        assert cl.dtw_distance(rs1, rs2, mode='abs') == rs1.nvalues
        assert cl.dtw_distance(rs1, rs2, mode='square') == np.sqrt(rs1.nvalues)

    def test_order_irrelevance_contant_ts(self):
        # Build two constant time series
        rs1 = ts.constant(start_date="2000-01-01", end_date="2020-01-01", frequency='Y', cst=1.)
        rs2 = ts.constant(start_date="2000-01-01", end_date="2020-01-01", frequency='Y', cst=2.)
        
        # Test that order of time series is irrelevant
        assert cl.euclidean_distance(rs1, rs2) == cl.euclidean_distance(rs1, rs2)
        assert cl.dtw_distance(rs1, rs2) == cl.dtw_distance(rs1, rs2)

    def test_order_irrelevance_random_ts(self):
        # Build two random series
        rs1 = ts.auto_regressive(start_date="2000-01-01", end_date="2020-01-01", frequency='M',
                                 start_values=[1.], cst=1., order=1, coeffs=[0.3], sigma=0.1)

        rs2 = ts.auto_regressive(start_date="2000-01-01", end_date="2020-01-01", frequency='M',
                                 start_values=[1.], cst=1., order=1, coeffs=[0.35], sigma=0.1)

        # Test that order of time series is irrelevant
        assert cl.euclidean_distance(rs1, rs2) == cl.euclidean_distance(rs1, rs2)
        assert cl.dtw_distance(rs1, rs2) == cl.dtw_distance(rs1, rs2)

    

