# Created on 2020/7/16

# This module is for functions generating random time series.

# Standard library imports
from datetime import datetime
from typing import Union

# Third party imports
import numpy as np
import pandas as pd
from typeguard import typechecked

# Local application imports
from .. import timeseries as ts


#---------#---------#---------#---------#---------#---------#---------#---------#---------#


### TIME SERIES MODELS ###


# Simple models

@typechecked
def constant(start_date: Union[str, datetime.date],
             end_date: Union[str, datetime.date],
             frequency: str,
             cst: float=0.,
             sigma: float=0.,
             tz=None, unit=None, name=""
             ) -> ts.TimeSeries:
    """
    Defines a time series with constant numerical value
    and eventually add a noise to it.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    cst : int or float
      The constant to build the time series from.
    sigma : float
      Standard deviation for the Gaussian noise.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.

    Returns
    -------
    TimeSeries
      The constant time series with eventual Gaussian noise.
    
    Raises
    ------
      None
    
    Notes
    -----
     White noise is Gaussian here.
     For offset aliases available see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate data
    if float(sigma) != 0.:
        rand_val = np.random.normal(loc=0., scale=sigma, size=T)
        data_vals = [cst] * T + rand_val
    else:
        data_vals = [cst] * T

    # Make time series
    df = pd.DataFrame(index=data_index, data=data_vals)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)

    return rs



# These models describe the evolution of time series.

@typechecked
def auto_regressive(start_date: Union[str, datetime.date],
                    end_date: Union[str, datetime.date],
                    frequency: str,
                    start_values: list,
                    cst: float,
                    order: int,
                    coeffs: list,
                    sigma: float,
                    tz: str=None,
                    unit: str=None,
                    name: str="",
                    verbose: bool=False
                    ) -> ts.TimeSeries:
    """
    Generates a time series from the Auto-Regressive (AR) model of arbitrary order P.
    
    The model is of the form:
    x_t = cst + coeffs[0] * x_{t-1} + ... + coeffs[P-1] * x_{t-P} + a_t
    where a_t is the white noise with standard deviation sigma.
    
    Initial values for {x_0, ..., x_P} are imposed from the values in start_values.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    start_values : list
      Initial values of the process (P of them).
    cst : float
      Constant value of the process.
    order : int
      Order of the process (i.e. value of P).
    coeffs : list
      Coefficients of the process.
    sigma : float
      Standard deviation of the Gaussian white noise.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
    verbose : bool
      Verbose option.
    
    Returns
    -------
    TimeSeries
      The time series resulting from the Auto-Regressive process.
    
    Raises
    ------
      None
    
    Notes
    -----
     White noise is Gaussian here.
     For offset aliases available see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Checks
    assert(len(coeffs)==order)
    assert(len(start_values)==order)
    P = len(start_values)
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the white noise (Note: p first values are not used)
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generate the random series
    x = [0.] * T
    for t_ini in range(P):
        x[t_ini] = start_values[t_ini]
    for t in range(P,T,1):
        x[t] = cst + a[t]
        for p in range(P):
            x[t] += coeffs[p] * x[t-p-1]
    
    # Compute theoretical expectation value
    if verbose:
        E = cst / (1 - sum(coeffs))
        print("Under stationarity assumption, the expected value for this AR({str(P)}) model is: {str(E)} \n")
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs


@typechecked
def random_walk(start_date: Union[str, datetime.date],
                end_date: Union[str, datetime.date],
                frequency: str,
                start_value: float,
                sigma: float,
                tz: str=None,
                unit: str=None,
                name: str=""
                ) -> ts.TimeSeries:
    """
    Generates a time series from the Random Walk process,
    i.e. an AR(1) model with {cst = 0, coeff[0] = 1}.
    
    The model is of the form:
    x_t = x_{t-1} + a_t
    where a_t is the white noise with standard deviation sigma.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    start_value : float
      Initial value of the process.
    sigma : float
      Standard deviation of the Gaussian white noise.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
    
    Returns
    -------
    TimeSeries
      The time series resulting from the Random Walk process.
    
    Raises
    ------
      None
    
    Notes
    -----
     White noise is Gaussian here.
     For offset aliases available see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the white noise (Note: first value is not used)
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generate the random series
    x = [0.] * T
    x[0] = start_value
    for t in range(1,T,1):
        x[t] = x[t-1] + a[t]
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs


@typechecked
def drift_random_walk(start_date: Union[str, datetime.date],
                      end_date: Union[str, datetime.date],
                      frequency: str,
                      start_value: float,
                      drift: float,
                      sigma: float,
                      tz: str=None,
                      unit: str=None,
                      name: str=""
                      ) -> ts.TimeSeries:
    """
    Generates a time series from the Random Walk with Drift process,
    i.e. an AR(1) model with {cst != 0, coeffs[0] = 1}.
    
    The model is of the form:
    x_t = drift + x_{t-1} + a_t
    where a_t is the white noise with standard deviation sigma.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    start_value : float
      Initial value of the process.
    drift : float
      Value of the drift.
    sigma : float
      Standard deviation of the Gaussian white noise.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
    
    Returns
    -------
    TimeSeries
      The time series resulting from the Random Walk process with drift.
    
    Raises
    ------
      None
    
    Notes
    -----
     White noise is Gaussian here.
     For offset aliases available see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the white noise (Note: first value is not used)
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generate the random series
    x = [0.] * T
    x[0] = start_value
    for t in range(1,T,1):
        x[t] = drift + x[t-1] + a[t]
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs


@typechecked
def moving_average(start_date: Union[str, datetime.date],
                   end_date: Union[str, datetime.date],
                   frequency: str,
                   cst: float,
                   order: int,
                   coeffs: list,
                   sigma: float,
                   tz: str=None,
                   unit: str=None,
                   name: str="",
                   verbose: bool=False
                   ) -> ts.TimeSeries:
    """
    Generates a time series from the Moving Average (MA) model of arbitrary order Q.
    
    The model is of the form:
    x_t = cst + a_t + coeffs[0] * a_{t-1} + ... + coeffs[Q-1] * a_{t-Q}
    where {a_t} is the white noise series with standard deviation sigma.

    We don't need to impose any initial values for {x_t}, they are imposed directly from {a_t}.
    
    To be clear, the initial steps of the process are:
    x_0 = cst + a_0
    x_1 = cst + a_1 + coeffs[0] * a_0
    x_2 = cst + a_2 + coeffs[0] * a_1 + coeffs[1] * a_0
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    cst : float
      Constant value of the process.
    order : int
      Order of the process (i.e. value of Q).
    coeffs : list
      List of coefficients.
    sigma : float
      Standard deviation of the Gaussian white noise.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
    verbose : bool
      Verbose option.
    
    Returns
    -------
    TimeSeries
      The time series resulting from the Moving Average process.
    
    Raises
    ------
      None
    
    Notes
    -----
     White noise is Gaussian here.
     For offset aliases available see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Checks
    assert(len(coeffs)==order)
    Q = order
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the white noise
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generate the random series
    x = [0.] * T
    for t in range(T):
        x[t] = cst + a[t]
        for q in range(Q):
            if t-q > 0:
                x[t] -= coeffs[q] * a[t-q-1]
    
    # Compute theoretical values
    if verbose:
        V = 1.
        for q in range(Q):
            V += coeffs[q]**2
        V *= sigma**2

        print(f"The expected value for this MA({str(Q)}) model is: {str(cst)}")
        print(f"The estimation of the variance for this MA({str(Q)}) model is: {str(V)}" + \
              f" , i.e. a standard deviation of: {str(np.sqrt(V))} \n")
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs


@typechecked
def arma(start_date: Union[str, datetime.date],
         end_date: Union[str, datetime.date],
         frequency: str,
         start_values: list,
         cst: float,
         ARorder: int,
         ARcoeffs: list,
         MAorder: int,
         MAcoeffs: list,
         sigma: float,
         tz: str=None,
         unit: str=None,
         name: str=""
         ) -> ts.TimeSeries:
    """
    Function generating a time series from the Auto-Regressive Moving Average (ARMA)
    model of orders (P,Q).
    
    The model is of the form:
    x_t = cst + Sum_{i=0}^{P-1} ARcoeffs[i] * a_{t-i-1}
        + a_t + Sum_{j=0}^{Q-1} MAcoeffs[j] * a_{t-j-1}
    where {a_t} is the white noise series with standard deviation sigma.
    
    Initial values for {x_0, ..., x_P} are imposed from the values in start_values.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    start_values : list
      Initial values of the process (P of them).
    cst : float
      Constant value of the process.
    ARorder : int
      Order of the AR part of the process (i.e. value of P).
    ARcoeffs : list
      List of coefficients for the AR part of the process.
    MAorder : int
      Order of the MA part of the process (i.e. value of Q).
    MAcoeffs : list
      List of coefficients for the MA part of the process.
    sigma : float
      Standard deviation of the Gaussian white noise.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
    
    Returns
    -------
    TimeSeries
      The time series resulting from the ARMA process.
    
    Raises
    ------
      None
    
    Notes
    -----
     White noise is Gaussian here.
     For offset aliases available see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Checks
    assert(len(ARcoeffs)==ARorder)
    assert(len(MAcoeffs)==MAorder)
    assert(len(start_values)==ARorder)
    P = ARorder
    Q = MAorder
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the white noise
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generate the random series
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
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=x)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs


@typechecked
def rca(start_date: Union[str, datetime.date],
        end_date: Union[str, datetime.date],
        frequency: str,
        cst: float,
        order: int,
        ARcoeffs: list,
        cov_matrix: list,
        sigma: float,
        tz: str=None,
        unit: str=None,
        name: str=""
        ) -> ts.TimeSeries:
    """
    Function generating a time series from the Random Coefficient Auto-Regressive (RCA)
    model of order M.
    
    The model is of the form:
    x_t = cst + Sum_{m=0}^{M-1} (ARcoeffs[m] + coeffs[m]) * x_{t-m-1} + a_t.
    
    Here {a_t} is a Gaussian white noise with standard deviation sigma
    and coeffs_t are randomly generated from the covariance matrix cov_matrix.
    
    In addition, we have some imposed coefficients of the Auto-Regressive type in ARcoeffs.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    cst : float
      Constant value of the process.
    order : int
      Order of the process (i.e. value of M).
    ARcoeffs : list
      List of coefficients for the AR part of the process.
    cov_matrix: list of lists
      Covariance matrix for the random part of the process.
    sigma : float
      Standard deviation of the Gaussian white noise.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
      
    Returns
    -------
    TimeSeries
      The time series resulting from the RCA process.
    
    Raises
    ------
      None
    
    Notes
    -----
     We assume coeffs_t follow a multivariate Gaussian distribution.
     Also cov_matrix should be a non-negative definite matrix.
     
     Here we do not have an argument called start_value,
     compared with randomseries.auto_regressive().
     This choice is made as there are already random coefficients involved.
     There is no real use in imposing the first values of the time series
     other than just ARcoeffs and the generated coeffs.
     
     For offset aliases available see:
     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Checks
    assert(len(ARcoeffs)==order)
    assert(len(cov_matrix)==order and len(cov_matrix[0])==order)
    for row in cov_matrix:
        for x in row:
            assert(x>=0)
    M = order
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the white noise
    a = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generate the random series
    x = [0.] * T
    for t in range(T):
        x[t] = cst + a[t]
        # Generate the list of coefficients
        coeffs = np.random.multivariate_normal(mean=[0.] * M, cov=cov_matrix, size=1)[0]
        for m in range(M):
            if t-m > 0:
                x[t] += (ARcoeffs[m] + coeffs[m]) * a[t-m-1]
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=a)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs



### HETEROSCEDASTIC MODELS ###

# These models describe the volatility of a time series.

@typechecked
def arch(start_date: Union[str, datetime.date],
         end_date: Union[str, datetime.date],
         frequency: str,
         cst: float,
         order: int,
         coeffs: list,
         tz: str=None,
         unit: str=None,
         name: str="",
         verbose: bool=False
         ) -> ts.TimeSeries:
    """
    Function generating a volatility series from the
    Auto-Regressive Conditional Heteroscedastic (ARCH) model of order M.
    
    The model is of the form:
    a_t = sig_t * eps_t
    with sig_t^2 = cst + coeffs[0] * a_{t-1}^2 + ... + coeffs[M-1] * a_{t-M}^2.
    
    Here {eps_t} is a sequence of idd random variables with mean zero and unit variance,
    i.e. a white noise with unit variance.
    
    The coefficients cst and coeffs[i] are assumed to be positive
    and must be such that a_t is finite with positive variance.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    cst : float
      Constant value of the process.
    order : int
      Order of the process (i.e. value of M).
    coeffs : list
      List of coefficients of the process.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
    verbose : bool
      Verbose option.
      
    Returns
    -------
    TimeSeries
      The time series resulting from the ARCH process.
    
    Raises
    ------
      None
    
    Notes
    -----
      For offset aliases available see:
      https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Checks
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
        print("Sum of coefficients must be < 1 in order to have positive variance.")
    assert(sum(coeffs)<1)
    M = order
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the "unit" white noise
    eps = np.random.normal(loc=0., scale=1, size=T)

    # Generate the random series
    a = [0.] * T
    for t in range(T):
        sig_square = cst
        for m in range(M):
            if t-m > 0:
                sig_square += coeffs[m] * a[t-m-1]**2
        sig = np.sqrt(sig_square)
        a[t] = sig * eps[t]
    
    # Compute theoretical values
    if verbose:
        print(f"The expected value for this ARCH({str(M)}) model is 0, like any other ARCH model," \
              + f" and the estimated value is: {str(np.mean(a))}")
        V = cst / (1 - sum(coeffs))
        print(f"The theoretical standard deviation value for this ARCH({str(M)}) model is: {str(V)}")
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=a)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs


@typechecked
def garch(start_date: Union[str, datetime.date],
          end_date: Union[str, datetime.date],
          frequency: str,
          cst: float,
          order_a: int,
          coeffs_a: list,
          order_sig: int,
          coeffs_sig: list,
          tz: str=None,
          unit: str=None,
          name: str="",
          verbose: bool=False
          ) -> ts.TimeSeries:
    """
    Function generating a volatility series from the
    Generalized ARCH (GARCH) model of order M.
    
    The model is of the form:
    a_t = sig_t * eps_t
    with sig_t^2 = cst + Sum_{i=0}^{M-1} coeffs_a[i] * a_{t-i-1}^2 
                       + Sum_{j=0}^{S-1} coeffs_sig[j] * sig_{t-j-1}^2.
                       
    Here {eps_t} is a sequence of idd random variables with mean zero and unit variance,
    i.e. a white noise with unit variance.
    
    The coefficients cst and coeffs[i] are assumed to be positive
    and must be such that a_t is finite with positive variance.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    cst : float
      Constant value of the process.
    order_a : int
      Order of the a_t part of the process (i.e. value of M).
    coeffs_a : list
      List of coefficients of the a_t part of the process.
    order_sig : int
      Order of the sig_t part of the process (i.e. value of S).
    coeffs_sig : list
      List of coefficients of the sig_t part of the process.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
    verbose : bool
      Verbose option.
      
    Returns
    -------
    TimeSeries
      The time series resulting from the GARCH process.
    
    Raises
    ------
      None
    
    Notes
    -----
      For offset aliases available see:
      https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Checks
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
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the "unit" white noise
    eps = np.random.normal(loc=0., scale=1, size=T)

    # Generate the random series
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

    # Compute theoretical values
    if verbose:
        V = cst / (1 - sum(coeffs_a) - sum(coeffs_sig))
        print(f"The theoretical standard deviation for this GARCH({str(M)}, {str(S)}) model is: {str(np.sqrt(V))}")
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=a)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs


@typechecked
def charma(start_date: Union[str, datetime.date],
           end_date: Union[str, datetime.date],
           frequency: str,
           order: int,
           cov_matrix: list,
           sigma: float,
           tz: str=None,
           unit: str=None,
           name: str=""
           ) -> ts.TimeSeries:
    """
    Function generating a volatility series from the
    Conditional Heterescedastic ARMA (CHARMA) model of order M.
    
    The model is of the form:
    a_t = Sum_{m=0}^{M-1} coeffs[m] * a_{t-m-1} + eta_t.
    
    Here {eta_t} is a Gaussian white noise with standard deviation sigma
    and coeffs_t are generated from the covariance matrix cov_matrix.
    
    Parameters
    ----------
    start_date : str or datetime
      Starting date of the time series.
    end_date : str or datetime
      Ending date of the time series.
    frequency : str or DateOffset
      Indicates the frequency of data as an offset alias (e.g. 'D' for days, 'M' for months, etc.).
    order : int
      Order of the process (i.e. value of M).
    cov_matrix: list of lists
      Covariance matrix for the random part of the process.
    sigma : float
      Standard deviation of the Gaussian white noise.
    tz : str
      Timezone name.
    unit : str or None
      Unit of the time series values.
    name : str
      Name or nickname of the series.
      
    Returns
    -------
    TimeSeries
      The time series resulting from the CHARMA process.
    
    Raises
    ------
      None
    
    Notes
    -----
      White noise is Gaussian here.
      
      We assume coeffs_t follow a multivariate Gaussian distribution.
      Also cov_matrix should be a non-negative definite matrix.
    
      For offset aliases available see:
      https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
    
    Examples
    --------
      None
    """
    
    # Checks
    assert(len(cov_matrix)==order and len(cov_matrix[0])==order)
    for row in cov_matrix:
        for x in row:
            assert(x>=0)
    M = order
    
    # Generate index
    data_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    T = len(data_index)
    
    # Generate the "unit" white noise
    eta = np.random.normal(loc=0., scale=sigma, size=T)
    
    # Generate the random series
    a = [0.] * T
    for t in range(T):
        a[t] = eta[t]
        # Generate the list of coefficients
        coeffs = np.random.multivariate_normal(mean=[0.] * M, cov=cov_matrix, size=1)[0]
        for m in range(M):
            if t-m > 0:
                a[t] += coeffs[m] * a[t-m-1]
    
    # Combine them into a time series
    df = pd.DataFrame(index=data_index, data=a)
    rs = ts.TimeSeries(df, tz=tz, unit=unit, name=name)
    
    return rs

    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#