# Created on 2020/7/16

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import timeseries.timeseries as ts


def AutoRegressive(start_date, end_date, frequency, start_values, coeffs, order, sigma):
    """
    Function generating a time series from the AutoRegressive (AR) model of an arbitrary order P.
    """
    assert(len(coeffs)==order+1)
    assert(len(start_values)==order)
    P = len(start_values)
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise (Note: p first values are not used)
    eps = np.random.normal(loc=0., scale=sigma, size=(T,1))
    
    # Generating the random series
    data_values = [0.] * T
    for t_ini in range(P):
        data_values[t_ini] = start_values[t_ini]
    for t in range(P,T,1):
        data_values[t] = coeffs[0]
        for p in range(1,P+1,1):
            data_values[t] += coeffs[p] * data_values[t-p]
        data_values[t] += eps[t][0]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=data_values)
    rs = ts.timeseries(df)
    return rs
    



