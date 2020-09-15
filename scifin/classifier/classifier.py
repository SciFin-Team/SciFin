# Created on 2020/8/10

# This module is for analysing and classifying time series and other objects.

# Standard library imports
from datetime import datetime
from datetime import timedelta
import itertools
from typing import Union
import random as random


# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss
from sklearn.metrics import silhouette_samples
from sklearn.model_selection._split import KFold
from typeguard import typechecked

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

@typechecked
def kmeans_base_clustering(corr: Union[np.ndarray, pd.DataFrame],
                           names_features: list=None,
                           max_num_clusters: int=10,
                           **kwargs
                           ) -> (pd.DataFrame, dict, pd.Series):
    """
    Perform base clustering with Kmeans.

    Arguments
    ---------
    corr: numpy.array or pd.DataFrame
      Correlation matrix.
    names_features : list of str
      List of names for features.
    max_num_clusters: int
      Maximum number of clusters.
    **kwargs
        Arbitrary keyword arguments for sklearn.cluster.KMeans().

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

      To learn more about sklearn.cluster.KMeans():
      https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """

    # Checks
    if not isinstance(max_num_clusters, int):
        raise AssertionError("max_num_clusters must be integer.")

    # Initializations
    corr = pd.DataFrame(data=corr, index=names_features, columns=names_features)
    silh_score = pd.Series()

    # Define the observations matrix
    Xobs = ( ((1 - corr.fillna(0))/2.)**.5 ).values

    # Modify it to get an Euclidean distance matrix
    X = np.zeros(shape=Xobs.shape)
    for i,j in itertools.product(range(X.shape[0]), range(X.shape[1])):
        X[i,j] = np.sqrt( sum((Xobs[i,:] - Xobs[j,:])**2) )
    X = pd.DataFrame(data=X, index=names_features, columns=names_features)

    # Loop to generate different numbers of clusters
    for i in range(2, max_num_clusters+1):

        # Define model and fit
        kmeans_current = KMeans(n_clusters=i, **kwargs).fit(X)

        # Compute silhouette score
        silh_current = silhouette_samples(X, kmeans_current.labels_)

        # Compute clustering quality q (t-statistic of silhouette score)
        quality_current = silh_current.mean()/silh_current.std()
        quality = silh_score.mean()/silh_score.std()

        # Keep best quality scores and clustering
        if np.isnan(quality) or (quality_current > quality):
            silh_score = silh_current
            kmeans = kmeans_current

    # Extract index according to sorted labels
    new_idx = np.argsort(kmeans.labels_)

    # Reorder rows and columns
    clustered_corr = corr.iloc[new_idx]
    clustered_corr = clustered_corr.iloc[:,new_idx]

    # Form clusters
    clusters = {i: clustered_corr.columns[np.where(kmeans.labels_==i)[0]].tolist()
                for i in np.unique(kmeans.labels_)}

    # Define a series with the silhouette score
    silh_score = pd.Series(silh_score, index=X.index)

    return clustered_corr, clusters, silh_score



@typechecked
def make_new_outputs(corr: Union[np.array, pd.DataFrame],
                     clusters: dict,
                     clusters2: dict
                     ) -> (pd.DataFrame, dict, pd.Series):
    """
    Makes new outputs for kmeans_advanced_clustering() by recombining two sets of clusters
    together, recomputing their correlation matrix, distance matrix, kmeans labels and silhouette scores.

    Arguments
    ---------
    corr : numpy.array or pd.DataFrame
      Correlation matrix.
    clusters : dict
      First set of clusters.
    clusters2 : dict
      Second set of clusters.

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

    # Initializations
    # Add clusters keys to the new cluster
    clusters_new = {}
    for i in clusters.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters[i])
    for i in clusters2.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters2[i])

    # Compute new correlation matrix
    new_idx = [j for i in clusters_new for j in clusters_new[i]]
    corr_new = corr.loc[new_idx, new_idx]

    # Compute the observation matrix
    Xobs = ( ((1-corr.fillna(0))/2.)**.5 ).values

    # Compute the Euclidean distance matrix
    X = np.zeros(shape=Xobs.shape)
    for i,j in itertools.product(range(X.shape[0]), range(X.shape[1])):
        X[i,j] = np.sqrt( sum((Xobs[i,:] - Xobs[j,:])**2) )
    new_names_features = corr_new.columns.tolist()
    X = pd.DataFrame(data=X, index=new_names_features, columns=new_names_features)

    # Add labels together
    kmeans_labels = np.zeros(len(X.columns))
    for i in clusters_new.keys():
        idxs = [X.index.get_loc(k) for k in clusters_new[i]]
        kmeans_labels[idxs] = i

    # Compute the silhouette scores
    silh_new = pd.Series(silhouette_samples(X, kmeans_labels), index=X.index)

    return corr_new, clusters_new, silh_new


@typechecked
def kmeans_advanced_clustering(corr: Union[np.ndarray, pd.DataFrame],
                               names_features: list=None,
                               max_num_clusters: int=None,
                               **kwargs
                               ) -> (pd.DataFrame, dict, pd.Series):
    """
    Perform advanced clustering with Kmeans.
    The base clustering is used first, then clusters quality is evaluated.
    For clusters whose quality is less than the averaged quality,
    the clustering is reran.
    
    Arguments
    ---------
    corr: numpy.array
      Correlation matrix.
    names_features : list of str
      List of names for features.
    max_num_clusters: int
      Maximum number of clusters.
    **kwargs
        Arbitrary keyword arguments for sklearn.cluster.KMeans().
    
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

      To learn more about sklearn.cluster.KMeans():
      https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    
    # Checks
    if (max_num_clusters is not None) and (not isinstance(max_num_clusters, int)):
        raise AssertionError("max_num_clusters must be integer.")
      
    # Initializations
    if max_num_clusters==None:
        max_num_clusters = corr.shape[1]-1
    if names_features is None:
        names_features = corr.columns.tolist()

    # Using base clustering as initial step
    corr1, clusters, silh = kmeans_base_clustering(corr,
                                                   names_features=names_features,
                                                   max_num_clusters=min(max_num_clusters, corr.shape[1]-1),
                                                   **kwargs)
    
    # Compute t-stat for each cluster
    cluster_tstats = {i: np.mean(silh[clusters[i]]) / np.std(silh[clusters[i]]) for i in clusters.keys()}

    # Obtain the mean t-stat over clusters
    print(cluster_tstats)
    tstat_mean = sum(cluster_tstats.values()) / len(cluster_tstats)

    # Select the clusters having a t-stat below the mean
    clusters_to_redo = [i for i in cluster_tstats.keys() if cluster_tstats[i] < tstat_mean]

    # Only one cluster to redo, nothing to do
    if len(clusters_to_redo)<=1:
        return corr1, clusters, silh

    # More than one cluster to redo
    else:
        # Get the key name of concerned clusters
        keys_redo = [j for i in clusters_to_redo for j in clusters[i]]

        # Compute their correlation
        corr_tmp = corr.loc[keys_redo, keys_redo]

        # Compute their mean t-stat
        tstat_mean = np.mean([cluster_tstats[i] for i in clusters_to_redo])

        # Redo the advanced clustering
        corr2, clusters2, silh2 = kmeans_advanced_clustering(corr_tmp,
                                                             max_num_clusters=min(max_num_clusters, corr_tmp.shape[1]-1),
                                                             **kwargs)
        # Make new outputs, if necessary
        clusters_not_redone = {i: clusters[i] for i in clusters.keys() if i not in clusters_to_redo}
        corr_new, clusters_new, silh_new = make_new_outputs(corr, clusters_not_redone, clusters2)

        # Compute the new t-stat mean
        new_tstat_mean = np.mean([ np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]])
                                   for i in clusters_new.keys() ])

        # Return the improved clusters (if improved)
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
        
        
def feature_importance_pvalues(fit, plot=False, figsize=(10,10)):
    """
    Plot the p-values of features from a fit.
    
    Arguments
    ---------
    fit : fit model
      Fit already apply on data.
    plot : bool
      Option to plot feature importance.
    figsize : (float, float)
      Dimensions of the plot.
    
    Returns
    -------
    pandas.DataFrame
      Data frame with features importance.
    """
    
    # Extract p-values and sort them
    pvals = fit.pvalues.sort_values(ascending=False)

    # Plot
    if plot is True:
        plt.figure(figsize=figsize)
        plt.title("P-values of features.")
        plt.barh(y=pvals.index, width=pvals.values)
        plt.show()
    
    # Make DataFrame
    pvals_df = pd.DataFrame(pvals)
    pvals_df.index.name = "Feature"
    pvals_df.columns = ["Importance"]
    
    return pvals_df


def feature_importance_mdi(classifier, X, y, plot=False, figsize=(10,10)):
    """
    Feature importance based on in-sample Mean-Decrease Impurity (MDI).
    
    Arguments
    ---------
    classifier : tree classifier
      Tree classifier to apply on data.
    X : pandas.DataFrame
      Data Frame with features as columns and samples as rows.
    y : pandas.Series
      Series containing class membership.
    plot : bool
      Option to plot feature importance.
    figsize : (float, float)
      Dimensions of the plot.
    
    Returns
    -------
    pandas.DataFrame
      Data frame with features importance.
      
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Checks
    if not isinstance(X, pd.DataFrame):
        raise AssertionError("X must be pandas.DataFrame.")
    if not isinstance(y, pd.Series):
        raise AssertionError("y must be pandas.Series.")
    
    # Fit
    fit = classifier.fit(X,y)
    
    # Extract feature importance
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    
    # Make Data Frame
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = X.columns
    df0 = df0.replace(0, np.nan)
    fimp_df = pd.concat( {'Importance Mean': df0.mean(), 'Importance Std': df0.std() * df0.shape[0]**(-0.5)}, axis=1)
    fimp_df /= fimp_df['Importance Mean'].sum()
    fimp_df.index.name = "Feature"
    
    # Sort values
    sorted_fimp = fimp_df.sort_values(by='Importance Mean')
    
    # Plot
    if plot is True:
        plt.figure(figsize=figsize)
        plt.title("Feature importance based on in-sample Mean-Decrease Impurity (MDI).")
        plt.barh(y=sorted_fimp.index, width=sorted_fimp['Importance Mean'], xerr=sorted_fimp['Importance Std'])
        plt.show()
    
    return fimp_df
        
    
def feature_importance_mda(classifier, X, y, n_splits=10, plot=False, figsize=(10,10)):
    """
    Feature importance based out-of-sample Mean-Decrease Accuracy (MDA).
    
    Arguments
    ---------
    classifier : tree classifier
      Tree classifier to apply on data.
    X : pandas.DataFrame
      Data Frame with features as columns and samples as rows.
    y : pandas.Series
      Series containing class membership.
    plot : bool
      Option to plot feature importance.
    figsize : (float, float)
      Dimensions of the plot.
    
    Returns
    -------
    pandas.DataFrame
      Data frame with features importance.
    
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Checks
    if not isinstance(X, pd.DataFrame):
        raise AssertionError("X must be pandas.DataFrame.")
    if not isinstance(y, pd.Series):
        raise AssertionError("y must be pandas.Series.")
    
    # Generate K-fold cross validation
    cv_gen = KFold(n_splits=n_splits)
    scr0 = pd.Series()
    scr1 = pd.DataFrame(columns=X.columns)
    
    # Loop over the folds:
    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        
        # Train/Test split
        X0, y0 = X.iloc[train,:], y.iloc[train]
        X1, y1 = X.iloc[test,:], y.iloc[test]
        
        # Fit
        fit = classifier.fit(X=X0, y=y0)
        
        # Prediction before shuffling
        prob = fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=classifier.classes_)
        
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            
            # Shuffle one column
            np.random.shuffle(X1_[j].values)
            
            # Prediction after shuffling
            prob = fit.predict_proba(X1_)
            scr1.loc[i,j] = -log_loss(y1, prob, labels=classifier.classes_)
    
    fimp_df = (-1*scr1).add(scr0, axis=0)
    fimp_df = fimp_df / (-1*scr1)
    fimp_df = pd.concat({'Importance Mean': fimp_df.mean(), 'Importance Std': fimp_df.std()*fimp_df.shape[0]**-.5}, axis=1)
    
    # Sort values
    sorted_fimp = fimp_df.sort_values(by='Importance Mean')
    
    # Plot
    if plot is True:
        plt.figure(figsize=figsize)
        plt.title("Feature importance based on out-of-sample Mean-Decrease Accuracy (MDA).")
        plt.barh(y=sorted_fimp.index, width=sorted_fimp['Importance Mean'], xerr=sorted_fimp['Importance Std'])
        plt.show()
    
    return sorted_fimp

        
#---------#---------#---------#---------#---------#---------#---------#---------#---------#