# __init__.py
__version__ = "0.1.0"
__author__ = "Fabien Nugier"

"""
The :mod:`scifin.timeseries` module includes methods for time series analysis.
"""

from .timeseries import Series, TimeSeries, CatTimeSeries, \
                        get_list_timezones, build_from_csv, build_from_dataframe, build_from_list, build_from_lists, \
                        multi_plot, multi_plot_distrib

from .randomseries import constant, auto_regressive, random_walk, drift_random_walk, moving_average, \
                          arma, rca, arch, garch, charma

