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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# Local application imports
from .. import timeseries


#---------#---------#---------#---------#---------#---------#---------#---------#---------#



def euclidean_distance(ts1, ts2):
    """
    Returns the Euclidean distance between two TimeSeries.
    
    Parameters
    ----------
    ts1 : TimeSeries
      First time series.
    ts2 : TimeSeries
      Second time series.
      
    Returns
    -------
    float
      Euclidean distance between time series.
    """
    
    # Checks
    try:
        assert(ts1.type=='TimeSeries' and ts2.type=='TimeSeries')
    except TypeError:
        raise TypeError("Series have to be of type TimeSeries.")
        
    try:
        assert(ts1.data.index.tolist() == ts2.data.index.tolist())
    except IndexError:
        raise IndexError("Time series do not have the same index.")
    
        
    # Return distance
    squares = (ts1.data - ts2.data)**2
    return np.sqrt(float(squares.sum()))
    
    
    
def dtw_distance(ts1, ts2, window=None):
    """
    Returns the Dynamic Time Warping (DTW) distance between two TimeSeries.
    A locality constraint can be used by specifying the size of a window.
    
    Parameters
    ----------
    ts1 : TimeSeries
      First time series.
    ts2 : TimeSeries
      Second time series.
    window : int
      Size of restrictive search window.
      
    Returns
    -------
    float
      DTW distance between time series.
      
    Notes
    -----
      To learn more about DTW, please refer to:
      https://en.wikipedia.org/wiki/Dynamic_time_warping
      
      Notice that taking a small window size may lead to
      a wrong estimate of the real dynamic time warping distance.
    """
    
    # Checks
    try:
        assert(ts1.type=='TimeSeries' and ts2.type=='TimeSeries')
    except TypeError:
        raise TypeError("Series have to be of type TimeSeries.")
        
    # Initializations
    N1 = len(ts1.data.index.tolist())
    N2 = len(ts2.data.index.tolist())
    if window is not None:
        assert(isinstance(window, int))
        w = window
    else:
        w = N2
    
    dtw = np.full(shape=(N1+1,N2+1), fill_value=np.inf)
    dtw[0,0] = 0

    # Loop
    for i in range(0, N1, 1):
        # for j in range(0, N2, 1):
        for j in range(max(0,int(i-w)), min(N2,int(i+w)), 1):
            square = (ts1.data.values[i] - ts2.data.values[j])**2
            dtw[i+1,j+1] = square + min(dtw[i,j+1], dtw[i+1,j], dtw[i,j])
            
    # Return distance
    return np.sqrt(dtw[N1, N2])


def kmeans_base_clustering(corr, max_num_clusters=10, n_init=10):
    """
    Perform the base clustering with Kmeans.
    
    
    Arguments
    ---------
    corr: numpy.array
      Correlation matrix.
    max_num_clusters: int
      Maximum number of clusters.
    n_init : int
      Initial value n_init for KMeans.
    
    Returns
    -------
    pd.DataFrame
      Clustered correlation matrix.
    dictionary
      List of clusters and their content.
    pd.Series
      Silhouette scores.
    
    Notes
    -----
      Function adapted from "Advances in Financial Machine Learning",
      Marcos López de Prado (2018).
    """
    
    # Checks
    if not isinstance(max_num_clusters, int):
        raise AssertionError("max_num_clusters must be integer.")
    if not isinstance(n_init, int):
        raise AssertionError("n_init must be integer.")
    
    # Initializations
    corr = pd.DataFrame(corr)
    silh_score = pd.Series()
    
    # Define the observations matrix X
    X = ((1 - corr.fillna(0))/2.)**.5
    
    # Loop to generate different initializations
    for init in range(n_init):
        
        # Loop to generate different numbers of clusters
        for i in range(2, max_num_clusters+1):
            
            # Define model and fit
            kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=n_init).fit(X)
            
            # Compute silhouette coefficients
            silh_ = silhouette_samples(X, kmeans_.labels_)
            
            # Compute clustering quality q (t-statistic of silhouette score)
            stat = (silh_.mean()/silh_.std(), silh_score.mean()/silh_score.std())
            if np.isnan(stat[1]) or (stat[0]>stat[1]):
                silh_score, kmeans = silh_, kmeans_
                
    # Extract index according to sorted labels
    new_idx = np.argsort(kmeans.labels_)
    
    # Reorder rows
    clustered_corr = corr.iloc[new_idx]
    # Reorder columns
    clustered_corr = clustered_corr.iloc[:,new_idx]
    
    # Form clusters
    clusters = {i: clustered_corr.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    
    # Define a series with the silhouette score
    silh_score = pd.Series(silh_score, index=X.index)

    return clustered_corr, clusters, silh_score


#---------#---------#---------#---------#---------#---------#---------#---------#---------#