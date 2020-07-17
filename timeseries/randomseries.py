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
    The model is of the form: x_t = coeff_0 + coeff_1 * x_{t-1} + ... + coeff_P * x_{t-P} + eps_t
    where eps_t is the white noise with standard deviation sigma.
    Initial values for {x_0, ..., x_P} are imposed from the values in start_values.
    """
    assert(len(coeffs)==order+1)
    assert(len(start_values)==order)
    P = len(start_values)
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise (Note: p first values are not used)
    eps = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    data_values = [0.] * T
    for t_ini in range(P):
        data_values[t_ini] = start_values[t_ini]
    for t in range(P,T,1):
        data_values[t] = coeffs[0]
        for p in range(1,P+1,1):
            data_values[t] += coeffs[p] * data_values[t-p]
        data_values[t] += eps[t]
    
    # Computing theoretical expectation value
    SumCoeff1toP = sum(coeffs) - coeffs[0]
    E = coeffs[0] / (1 - SumCoeff1toP)
    print("Under stationarity assumption, the expectation value for this AR(" + str(P) + ") model is: " + str(E) + "\n")
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=data_values)
    rs = ts.timeseries(df)
    return rs
    



def MovingAverage(start_date, end_date, frequency, coeffs, order, sigma):
    """
    Function generating a time series from the MovingAverage (MA) model of an arbitrary order Q.
    The model is of the form: x_t = coeff_0 + eps_t + coeff_1 * eps_{t-1} + ... + coeff_Q * eps_{t-Q}
    where {eps_t} is the white noise series with standard deviation sigma.
    We don't need to impose any initial values for {x_t} are imposed directly from {eps_t}.
    
    Clarification: We thus assume {x_0 = coeff_0 + eps_0 ; x_1 = coeff_0 + eps_1 + coeff_1 * eps_0 ; x_2 = coeff_0 + eps_2 + coeff_1 * eps_1 + coeff_2 * eps_0}.
    """
    assert(len(coeffs)==order+1)
    Q = order
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise (Note: p first values are not used)
    eps = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    data_values = [0.] * T
    for t in range(T):
        data_values[t] = coeffs[0] + eps[t]
        for q in range(1,Q+1,1):
            if t-q >= 0:
                data_values[t] += coeffs[q] * eps[t-q]
    
    # Computing theoretical values
    V = 1.
    for q in range(1,Q+1,1):
        V += coeffs[q]**2
    V *= sigma**2
    print("The expectation value for this MA(" + str(Q) + ") model is: " + str(coeffs[0]))
    print("The estimation of the variance for this MA(" + str(Q) + ") model is: " + str(V) + \
          " , i.e. a standard deviation of: " + str(np.sqrt(V)) + "\n")
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=data_values)
    rs = ts.timeseries(df)
    return rs