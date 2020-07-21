# Created on 2020/7/16

# This module is for functions generating random time series.

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import timeseries.timeseries as ts



### TIME SERIES MODELS ###

# These models describe the evolution of time series.


def AutoRegressive(start_date, end_date, frequency, start_values, cst, order, coeffs, sigma):
    """
    Function generating a time series from the Auto-Regressive (AR) model of an arbitrary order P.
    The model is of the form: x_t = cst + coeffs[0] * x_{t-1} + ... + coeffs[P-1] * x_{t-P} + a_t
    where a_t is the white noise with standard deviation sigma.
    Initial values for {x_0, ..., x_P} are imposed from the values in start_values.
    
    Note: noise is Gaussian here.
    """
    assert(len(coeffs)==order)
    assert(len(start_values)==order)
    P = len(start_values)
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise (Note: p first values are not used)
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    x = [0.] * T
    for t_ini in range(P):
        x[t_ini] = start_values[t_ini]
    for t in range(P,T,1):
        x[t] = cst + a[t]
        for p in range(P):
            x[t] += coeffs[p] * x[t-p-1]
    
    # Computing theoretical expectation value
    E = cst / (1 - sum(coeffs))
    print("Under stationarity assumption, the expected value for this AR(" + str(P) + ") model is: " + str(E) + "\n")
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.timeseries(df)
    return rs
    


def RandomWalk(start_date, end_date, frequency, start_value, sigma):
    """
    Function generating a time series from the Random Walk process, i.e. an AR(1) model with {cst = 0, coeff[0] = 1}.
    The model is of the form: x_t = x_{t-1} + a_t where a_t is the white noise with standard deviation sigma.
    
    Note: noise is Gaussian here.
    """
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise (Note: first value is not used)
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    x = [0.] * T
    x[0] = start_value
    for t in range(1,T,1):
        x[t] = x[t-1] + a[t]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.timeseries(df)
    return rs
    

    
def DriftRandomWalk(start_date, end_date, frequency, start_value, drift, sigma):
    """
    Function generating a time series from the Random Walk with Drift process, i.e. an AR(1) model with {cst != 0, coeffs[0] = 1}.
    The model is of the form: x_t = drift + x_{t-1} + a_t where a_t is the white noise with standard deviation sigma.
    
    Note: noise is Gaussian here.
    """
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise (Note: first value is not used)
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    x = [0.] * T
    x[0] = start_value
    for t in range(1,T,1):
        x[t] = drift + x[t-1] + a[t]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.timeseries(df)
    return rs
    
    

def MovingAverage(start_date, end_date, frequency, cst, order, coeffs, sigma):
    """
    Function generating a time series from the Moving Average (MA) model of an arbitrary order Q.
    The model is of the form: x_t = cst + a_t + coeffs[0] * a_{t-1} + ... + coeffs[Q-1] * a_{t-Q}
    where {a_t} is the white noise series with standard deviation sigma.
    We don't need to impose any initial values for {x_t} are imposed directly from {a_t}.
    
    Clarification: We thus assume {x_0 = cst + a_0 ; x_1 = cst + a_1 + coeffs[0] * a_0 ;
    x_2 = cst + a_2 + coeffs[0] * a_1 + coeffs[1] * a_0} ; ...
    
    Note: noise is Gaussian here.
    """
    assert(len(coeffs)==order)
    Q = order
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    x = [0.] * T
    for t in range(T):
        x[t] = cst + a[t]
        for q in range(Q):
            if t-q > 0:
                x[t] -= coeffs[q] * a[t-q-1]
    
    # Computing theoretical values
    V = 1.
    for q in range(Q):
        V += coeffs[q]**2
    V *= sigma**2
    print("The expected value for this MA(" + str(Q) + ") model is: " + str(cst))
    print("The estimation of the variance for this MA(" + str(Q) + ") model is: " + str(V) + \
          " , i.e. a standard deviation of: " + str(np.sqrt(V)) + "\n")
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.timeseries(df)
    return rs



def ARMA(start_date, end_date, frequency, start_values, cst, ARorder, ARcoeffs, MAorder, MAcoeffs, sigma):
    """
    Function generating a time series from the Auto-Regressive Moving Average (ARMA) model of orders (P,Q).
    The model is of the form: x_t = cst + Sum_{i=0}^{P-1} ARcoeffs[i] * a_{t-i-1} + a_t + Sum_{j=0}^{Q-1} MAcoeffs[j] * a_{t-j-1}
    where {a_t} is the white noise series with standard deviation sigma.
    Initial values for {x_0, ..., x_P} are imposed from the values in start_values.
    
    Note: noise is Gaussian here.
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
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    x = [0.] * T
    # Taking care of {x_0, x_1, ..., x_P}
    for t_ini in range(P):
        x[t_ini] = start_values[t_ini]
    # Taking care of the rest
    for t in range(P,T,1):
        x[t] = cst + a[t]
        for p in range(P):
            x[t] += ARcoeffs[p] * x[t-p]
        for q in range(Q):
            if t-q > 0:
                x[t] -= MAcoeffs[q] * x[t-q-1]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.timeseries(df)
    return rs



def RCA(start_date, end_date, frequency, cst, order, ARcoeffs, cov_matrix, sigma):
    """
    Function generating a time series from the Random Coefficient Auto-Regressive (RCA) model of order M.
    The model is of the form: x_t = cst + Sum_{m=0}^{M-1} (ARcoeffs[m] + coeffs[m]) * x_{t-m-1} + a_t.
    Here {a_t} is a Gaussian white noise with standard deviation sigma and coeffs_t are randomly generated from the covariance matrix cov_matrix.
    In addition, we have some imposed coefficients of the Auto-Regressive type in ARcoeffs.
    
    Note: we assume coeffs_t follow a multivariate Gaussian distribution. Also cov_matrix should be a non-negative definite matrix.
    Note: here we do not have an argument called start_value, compared to randomseries.AutoRegressive(). This choice is made as there are already random coefficients
    involved, here there is no real use in imposing the first values of the time series other than just ARcoeffs and the generated coeffs.
    """
    # Checks
    assert(len(ARcoeffs)==order)
    assert(len(cov_matrix)==order and len(cov_matrix[0])==order)
    for row in cov_matrix:
        for x in row:
            assert(x>=0)
    M = order
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the white noise
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    x = [0.] * T
    for t in range(T):
        x[t] = cst + a[t]
        # Generating the list of coefficients
        coeffs = np.random.multivariate_normal(mean=[0.] * M, cov=cov_matrix, size=1)[0]
        for m in range(M):
            if t-m > 0:
                x[t] += (ARcoeffs[m] + coeffs[m]) * a[t-m-1]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=a)
    rs = ts.timeseries(df)
    return rs






### HETEROSCEDASTIC MODELS ###

# These models describe the volatility of a time series.

def ARCH(start_date, end_date, frequency, cst, order, coeffs):
    """
    Function generating a volatility series from the Auto-Regressive Conditional Heteroscedastic (ARCH) model of order M.
    The model is of the form: a_t = sig_t * eps_t
    with sig_t^2 = cst + coeffs[0] * a_{t-1}^2 + ... + coeffs[M-1] * a_{t-M}^2.
    Here {eps_t} is a sequence of idd random variables with mean zero and unit variance, i.e. a white noise with unit variance.
    The coefficients cst and coeffs[i] are assumed to be positive and must be such that a_t is finite with positive variance.
    """
    # Conditions
    assert(len(coeffs)==order)
    # Non-negativity
    if(cst<=0):
        print("cst must be strictly positive.")
    assert(cst>0)
    for x in coeffs:
        if (x<0):
            print("coefficients are not allowed to take negative values.")
        assert(x>=0)
    # Sum less than unity
    if(sum(coeffs)>=1):
        print("Sum of coefficients must be less than one in order to have positive variance.")
    assert(sum(coeffs)<1)
    M = order
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the "unit" white noise
    eps = np.random.normal(loc=0., scale=1, size=T)

    # Generating the random series
    a = [0.] * T
    for t in range(T):
        sig_square = cst
        for m in range(M):
            if t-m > 0:
                sig_square += coeffs[m] * a[t-m-1]**2
        sig = np.sqrt(sig_square)
        a[t] = sig * eps[t]
    
    # Computing theoretical values
    print("The expected value for this ARCH(" + str(M) + ") model is 0, like any other ARCH model, and the estimated value is : " + str(np.mean(a)))
    V = cst / (1 - sum(coeffs))
    print("The theoretical standard deviation value for this ARCH(" + str(M) + ") model is: " + str(V))
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=a)
    rs = ts.timeseries(df)
    return rs



def GARCH(start_date, end_date, frequency, cst, order_a, coeffs_a, order_sig, coeffs_sig):
    """
    Function generating a volatility series from the Generalized ARCH (GARCH) model of order M.
    The model is of the form: a_t = sig_t * eps_t
    with sig_t^2 = cst + Sum_{i=0}^{M-1} coeffs_a[i] * a_{t-i-1}^2 + Sum_{j=0}^{S-1} coeffs_sig[j] * sig_{t-j-1}^2.
    Here {eps_t} is a sequence of idd random variables with mean zero and unit variance, i.e. a white noise with unit variance.
    The coefficients cst and coeffs[i] are assumed to be positive and must be such that a_t is finite with positive variance.
    """
    # Conditions
    assert(len(coeffs_a)==order_a)
    assert(len(coeffs_sig)==order_sig)
    # Non-negativity
    if(cst<=0):
        print("cst must be strictly positive.")
    assert(cst>0)
    for x in coeffs_a + coeffs_sig:
        if (x<0):
            print("coefficients are not allowed to take negative values.")
        assert(x>=0)
    # Sum less than unity
    if(sum(coeffs_a) + sum(coeffs_sig) >= 1):
        print("Sum of coefficients must be less than one in order to have positive variance.")
    assert(sum(coeffs_a) + sum(coeffs_sig) < 1)
    M = order_a
    S = order_sig
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the "unit" white noise
    eps = np.random.normal(loc=0., scale=1, size=T)

    # Generating the random series
    a = [0.] * T
    sig = [0.] * T
    for t in range(T):
        sig_square = cst
        for m in range(M):
            if t-m > 0:
                sig_square += coeffs_a[m] * a[t-m-1]**2
        for s in range(S):
            if t-s > 0:
                sig_square += coeffs_sig[s] * sig[t-s-1]**2
        sig[t] = np.sqrt(sig_square)
        a[t] = sig[t] * eps[t]

    # Computing theoretical values
    V = cst / (1 - sum(coeffs_a) - sum(coeffs_sig))
    print("The theoretical standard deviation for this GARCH(" + str(M) + "," + str(S) + ") model is: " + str(np.sqrt(V)))
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=a)
    rs = ts.timeseries(df)
    return rs



def CHARMA(start_date, end_date, frequency, order, cov_matrix, sigma):
    """
    Function generating a volatility series from the Conditional Heterescedastic ARMA (CHARMA) model of order M.
    The model is of the form: a_t = Sum_{m=0}^{M-1} coeffs[m] * a_{t-m-1} + eta_t.
    Here {eta_t} is a Gaussian white noise with standard deviation sigma and coeffs_t are generated from the covariance matrix cov_matrix.
    Note: we assume coeffs_t follow a multivariate Gaussian distribution. Also cov_matrix should be a non-negative definite matrix.
    """
    # Checks
    assert(len(cov_matrix)==order and len(cov_matrix[0])==order)
    for row in cov_matrix:
        for x in row:
            assert(x>=0)
    M = order
    
    # Generating index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generating the "unit" white noise
    eta = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generating the random series
    a = [0.] * T
    for t in range(T):
        a[t] = eta[t]
        # Generating the list of coefficients
        coeffs = np.random.multivariate_normal(mean=[0.] * M, cov=cov_matrix, size=1)[0]
        for m in range(M):
            if t-m > 0:
                a[t] += coeffs[m] * a[t-m-1]
    
    # Combining them into a time series
    df = pd.DataFrame(index=data_index, data=a)
    rs = ts.timeseries(df)
    return rs

    

