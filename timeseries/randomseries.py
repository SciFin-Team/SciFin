# Created on 2020/7/16

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import timeseries.timeseries as ts



### TIME SERIES MODELS ###

def AutoRegressive(start_date, end_date, frequency, start_values, cst, order, coeffs, sigma):
    """
    Function generating a time series from the Auto-Regressive (AR) model of an arbitrary order P.
    The model is of the form: x_t = cst + coeffs[0] * x_{t-1} + ... + coeffs[P-1] * x_{t-P} + eps_t
    where eps_t is the white noise with standard deviation sigma.
    Initial values for {x_0, ..., x_P} are imposed from the values in start_values.
    """
    assert(len(coeffs)==order)
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
        data_values[t] = cst + eps[t]
        for p in range(P):
            data_values[t] += coeffs[p] * data_values[t-p-1]
    
    # Computing theoretical expectation value
    E = cst / (1 - sum(coeffs))
    print("Under stationarity assumption, the expected value for this AR(" + str(P) + ") model is: " + str(E) + "\n")
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=data_values)
    rs = ts.timeseries(df)
    return rs
    


def RandomWalk(start_date, end_date, frequency, start_value, sigma):
    """
    Function generating a time series from the Random Walk process, i.e. an AR(1) model with {cst = 0, coeff[0] = 1}.
    The model is of the form: x_t = x_{t-1} + eps_t where eps_t is the white noise with standard deviation sigma.
    """
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise (Note: first value is not used)
    eps = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    data_values = [0.] * T
    data_values[0] = start_value
    for t in range(1,T,1):
        data_values[t] = data_values[t-1] + eps[t]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=data_values)
    rs = ts.timeseries(df)
    return rs
    

    
def DriftRandomWalk(start_date, end_date, frequency, start_value, drift, sigma):
    """
    Function generating a time series from the Random Walk with Drift process, i.e. an AR(1) model with {cst != 0, coeffs[0] = 1}.
    The model is of the form: x_t = drift + x_{t-1} + eps_t where eps_t is the white noise with standard deviation sigma.
    """
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise (Note: first value is not used)
    eps = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    data_values = [0.] * T
    data_values[0] = start_value
    for t in range(1,T,1):
        data_values[t] = drift + data_values[t-1] + eps[t]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=data_values)
    rs = ts.timeseries(df)
    return rs
    
    

def MovingAverage(start_date, end_date, frequency, cst, order, coeffs, sigma):
    """
    Function generating a time series from the Moving Average (MA) model of an arbitrary order Q.
    The model is of the form: x_t = cst + eps_t + coeffs[0] * eps_{t-1} + ... + coeffs[Q-1] * eps_{t-Q}
    where {eps_t} is the white noise series with standard deviation sigma.
    We don't need to impose any initial values for {x_t} are imposed directly from {eps_t}.
    
    Clarification: We thus assume {x_0 = cst + eps_0 ; x_1 = cst + eps_1 + coeffs[0] * eps_0 ;
    x_2 = cst + eps_2 + coeffs[0] * eps_1 + coeffs[1] * eps_0} ; ...
    """
    assert(len(coeffs)==order)
    Q = order
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise
    eps = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    data_values = [0.] * T
    for t in range(T):
        data_values[t] = cst + eps[t]
        for q in range(Q):
            if t-q-1 >= 0:
                data_values[t] -= coeffs[q] * eps[t-q-1]
    
    # Computing theoretical values
    V = 1.
    for q in range(Q):
        V += coeffs[q]**2
    V *= sigma**2
    print("The expected value for this MA(" + str(Q) + ") model is: " + str(cst))
    print("The estimation of the variance for this MA(" + str(Q) + ") model is: " + str(V) + \
          " , i.e. a standard deviation of: " + str(np.sqrt(V)) + "\n")
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=data_values)
    rs = ts.timeseries(df)
    return rs



def ARMA(start_date, end_date, frequency, start_values, cst, ARorder, ARcoeffs, MAorder, MAcoeffs, sigma):
    """
    Function generating a time series from the Auto-Regressive Moving Average (ARMA) model of orders (P,Q).
    The model is of the form: x_t = cst + Sum_{i=0}^{P-1} ARcoeffs[i] * eps_{t-i} + eps_t + Sum_{j=0}^{Q-1} MAcoeffs[j] * eps_{t-j}
    where {eps_t} is the white noise series with standard deviation sigma.
    Initial values for {x_0, ..., x_P} are imposed from the values in start_values.
    """
    assert(len(ARcoeffs)==ARorder)
    assert(len(MAcoeffs)==MAorder)
    assert(len(start_values)==ARorder)
    P = ARorder
    Q = MAorder
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise
    eps = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    data_values = [0.] * T
    # Taking care of {x_0, x_1, ..., x_P}
    for t_ini in range(P):
        data_values[t_ini] = start_values[t_ini]
    # Taking care of the rest
    for t in range(P,T,1):
        data_values[t] = cst + eps[t]
        for p in range(P):
            data_values[t] += ARcoeffs[p] * data_values[t-p]
        for q in range(Q):
            if t-q-1 >= 0:
                data_values[t] -= MAcoeffs[q] * data_values[t-q-1]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=data_values)
    rs = ts.timeseries(df)
    return rs




### HETEROSCEDASTIC MODELS ###




