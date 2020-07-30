# __init__.py
__version__ = "0.0.6"

"""
The :mod:`scifin.timeseries` module includes methods for time series analysis.
"""

from .timeseries import timeseries, multi_plot
from .randomseries import AutoRegressive, RandomWalk, DriftRandomWalk, MovingAverage, ARMA, RCA, \
                          ARCH, GARCH, CHARMA

