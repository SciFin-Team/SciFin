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
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# Local application imports
from .. import timeseries


#---------#---------#---------#---------#---------#---------#---------#---------#---------#

# DISTANCES

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


# KMEANS CLUSTERING

def kmeans_base_clustering(corr, names_features=None, max_num_clusters=10, n_init=10):
    """
    Perform base clustering with Kmeans.
    
    Arguments
    ---------
    corr: numpy.array
      Correlation matrix.
    names_features : list of str
      List of names for features.
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
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Checks
    if not isinstance(max_num_clusters, int):
        raise AssertionError("max_num_clusters must be integer.")
    if not isinstance(n_init, int):
        raise AssertionError("n_init must be integer.")
    
    # Initializations
    corr = pd.DataFrame(data=corr, index=names_features, columns=names_features)
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




def make_new_outputs(corr, clusters, clusters2):
    """
    Makes new outputs for kmeans_advanced_clustering().
    
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    clusters_new = {}
    for i in clusters.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters[i])
    for i in clusters2.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters2[i])
        
    new_idx = [j for i in clusters_new for j in clusters_new[i]] 
    corr_new = corr.loc[new_idx, new_idx]
    x = ((1-corr.fillna(0))/2.)**.5
    
    kmeans_labels = np.zeros(len(x.columns))
    for i in clusters_new.keys():
        idxs = [x.index.get_loc(k) for k in clusters_new[i]]
        kmeans_labels[idxs] = i

    silh_new = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)
    
    return corr_new, clusters_new, silh_new



def kmeans_advanced_clustering(corr, names_features=None, max_num_clusters=None, n_init=10):
    """
    Perform advanced clustering with Kmeans.
    The base clustering is used first, then clusters quality is evaluated.
    For clusters whose quality less than the averaged quality,
    the clustering is reran.
    
    Arguments
    ---------
    corr: numpy.array
      Correlation matrix.
    names_features : list of str
      List of names for features.
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
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Checks
    if (max_num_clusters is not None) and (not isinstance(max_num_clusters, int)):
        raise AssertionError("max_num_clusters must be integer.")
    if not isinstance(n_init, int):
        raise AssertionError("n_init must be integer.")
      
    # Initializations
    if max_num_clusters==None:
        max_num_clusters = corr.shape[1]-1
    # Using base clustering as initial step
    corr1, clusters, silh = kmeans_base_clustering(corr,
                                                   names_features=names_features,
                                                   max_num_clusters=min(max_num_clusters, corr.shape[1]-1),
                                                   n_init=n_init)
    
    # Compute t-stat for each cluster
    cluster_tstats = {i: np.mean(silh[clusters[i]]) / np.std(silh[clusters[i]]) for i in clusters.keys()}
    # Obtain the mean over clusters
    tstat_mean = sum(cluster_tstats.values()) / len(cluster_tstats)
    # Select the clusters which have a t-stat below the mean
    redo_clusters = [i for i in cluster_tstats.keys() if cluster_tstats[i] < tstat_mean]

    # Only one cluster
    if len(redo_clusters)<=1:
        return corr1, clusters, silh
    # More than one cluster
    else:
        keys_redo = [j for i in redo_clusters for j in clusters[i]]
        corr_tmp = corr.loc[keys_redo, keys_redo]
        tstat_mean = np.mean([cluster_tstats[i] for i in redo_clusters])
        corr2, clusters2, silh2 = kmeans_advanced_clustering(corr_tmp, max_num_clusters=min(max_num_clusters, corr_tmp.shape[1]-1), n_init=n_init)
        
        # Make new outputs, if necessary
        corr_new, clusters_new, silh_new = make_new_outputs(corr, {i: clusters[i] for i in clusters.keys() if i not in redo_clusters}, clusters2)
        new_tstat_mean = np.mean([ np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]]) for i in clusters_new.keys() ])
        
        if new_tstat_mean <= tstat_mean:
            return corr1, clusters, silh
        else:
            return corr_new, clusters_new, silh_new
        

        
# FEATURE IMPORTANCE

def generate_random_classification(n_features, n_informative, n_redundant, n_samples, random_state=0, sigma_std=0.):
    """
    Generate a random dataset for a classification problem.
    
    Arguments
    ---------
    n_features : int
      Total number of features.
    n_informative : int
      Number of informative features.
    n_redundant : int
      Number of redundant features.
    n_samples : int
      Number of samples.
    random_state : int
      See random state.
    sigma_std: float
      Standard deviation of added noise.
      
    Returns
    -------
    pandas.DataFrame
      Data Frame with features as columns and samples as rows.
    pandas.Series
      Series containing class membership.
    
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Checks
    for arg in [('n_features',n_features), ('n_informative',n_informative), ('n_redundant',n_redundant),
                ('n_samples',n_samples), ('random_state',random_state)]:
        if not isinstance(arg[1],int):
            raise AssertionError(arg[0] + " must be integer.")
    if not isinstance(arg[1], float) and not isinstance(arg[1], int):
        raise AssertionError("sigma_std must be float.")
    
    # Initializations
    np.random.seed(random_state)
    
    # Generate classification
    X, y = make_classification(n_samples = n_samples,
                               n_features = n_features - n_redundant,
                               n_informative = n_informative,
                               n_redundant = 0,
                               shuffle = False,
                               random_state = random_state)
    
    # Informed explanatory variables
    cols = ['I_' + str(i) for i in range(n_informative)]
    
    # Noise explanatory variables
    cols += ['N_' + str(i) for i in range(n_features - n_informative - n_redundant)]
    
    X = pd.DataFrame(X, columns=cols)
    y = pd.Series(y)
    i = np.random.choice(range(n_informative), size=n_redundant)
    
    # Redundant explanatory variables
    for k, j in enumerate(i):
        X['R_' + str(k)] = X['I_' + str(j)] + np.random.normal(size=X.shape[0]) * sigma_std
        
    return X, y
        
        
        
        
#---------#---------#---------#---------#---------#---------#---------#---------#---------#