# Created on 2020/8/10

# This module is for analysing and classifying time series and other objects.

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



def euclidean_distance(ts1, ts2):
    """
    Returns the Euclidean distance between two TimeSeries.
    """
    
    # Checks
    try:
        assert(ts1.data.index.tolist() == ts2.data.index.tolist())
    except IndexError:
        raise IndexError("Time series do not have the same index.")
    
    squares = (ts1.data - ts2.data)**2
    return float(squares.sum())
    
    
    






















#---------#---------#---------#---------#---------#---------#---------#---------#---------#