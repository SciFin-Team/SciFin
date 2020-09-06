# Created on 2020/8/11

# This module is for generating Monte Carlo simulations.

# Standard library imports
from datetime import datetime
from datetime import timedelta
import random as random

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local application imports
from .. import timeseries


#---------#---------#---------#---------#---------#---------#---------#---------#---------#

 
def generate_series(n=1, generate_names=True, series_model=None, **kwargs):
    """
    Generate a list of `n` series of the type `series_model`.
    Here all series have the same building parameters.
    
    Parameters
    ----------
    n : int
      Number of time series to be generated.
    series_model : function
      TimeSeries generating function.
    **kwargs
        Arbitrary keyword arguments.
      
    Returns
    -------
    List of TimeSeries
      The n time series that were generated.
    """
    
    # Checks
    assert(isinstance(n,int))

    # Create list
    L = []
    for i in range(n):
        if generate_names is True:
            L.append(series_model(name="rs"+str(i), **kwargs))
        else:
            L.append(series_model(**kwargs))
    
    return L



#---------#---------#---------#---------#---------#---------#---------#---------#---------#