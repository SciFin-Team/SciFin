# __init__.py
__version__ = "0.0.9"
__author__ = "Fabien Nugier"

"""
The :mod:`scifin.timeseries` module includes methods for time series analysis.
"""

from .timeseries import Series, TimeSeries, CatTimeSeries, \
                        build_from_csv, multi_plot, multi_plot_distrib

from .randomseries import constant, auto_regressive, random_walk, drift_random_walk, moving_average, \
                          ARMA, RCA, ARCH, GARCH, CHARMA

