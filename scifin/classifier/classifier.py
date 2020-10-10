# Created on 2020/8/10

# This module is for analysing and classifying time series and other objects.

# Standard library imports
import itertools
from typing import Any, Callable, Optional, Union
import multiprocessing as mp

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import cluster
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import silhouette_samples
from sklearn.model_selection._split import KFold
from typeguard import typechecked
import statsmodels.discrete.discrete_model

# Local application imports
from .. import timeseries as ts


#---------#---------#---------#---------#---------#---------#---------#---------#---------#

# DISTANCES

@typechecked
def euclidean_distance(ts1: ts.TimeSeries, ts2: ts.TimeSeries) -> float:
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
    
    
@typechecked
def dtw_distance(ts1: ts.TimeSeries,
                 ts2: ts.TimeSeries,
                 window: int=None,
                 mode: str='abs',
                 verbose: bool=False
                 ) -> float:
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
    mode : str
      Mode to choose among:
      - 'abs' for absolute value distance based calculation.
      - 'square' for squared value distance based calculation, with sqrt taken at the end.
      
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
    if mode not in ('abs', 'square'):
        raise AssertionError("Argument mode must be a string, either 'abs' or 'square'.")

    # Initializations
    # Window size
    N1 = len(ts1.data.index.tolist())
    N2 = len(ts2.data.index.tolist())
    if window is None:
        window = N2
    w = max(window, abs(N2-N1))

    # Prepare dtw matrix
    dtw = np.full(shape=(N1+1,N2+1), fill_value=np.inf)
    dtw[0,0] = 0
    for i in range(1, N1+1, 1):
        for j in range(max(1,i-w), min(N2,i+w)+1, 1):
            dtw[i,j] = 0

    # Loop
    for i in range(1, N1+1, 1):
        for j in range(max(1,i-w), min(N2,i+w)+1, 1):
            if mode=='abs':
                cost = abs(ts1.data.values[i-1] - ts2.data.values[j-1])
            elif mode=='square':
                cost = (ts1.data.values[i-1] - ts2.data.values[j-1])**2
            dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    if verbose:
        print(dtw)
            
    # Return distance
    if mode=='abs':
        return dtw[N1, N2]
    elif mode=='square':
        return np.sqrt(dtw[N1, N2])


#@typechecked
def distance_matrix_from_ts(list_ts, #: list,
                            distance_model=euclidean_distance, #: Callable[[], float],
                            normalize=False, #: bool = False,
                            **kwargs #: Any
                            ): # -> pd.DataFrame:
    """
    Computes the dtw distance between time series of a list.

    Parameters
    ----------
    list_ts : list
      List of time series.
    distance_model : function
      Model to compute the distance between time series.
    normalize : bool
      Option to normalize the matrix (values between -1 and 1)
    **kwargs :
      Extra arguments for the distance_model function.

    Returns
    -------
    pd.DataFrame
      DataFrame containing dtw-distances between time series.
    """

    # Checks
    N = len(list_ts)
    if N < 2:
        raise AssertionError("Argument list_ts must have at least 2 time series in it.")

    # Initialization
    list_names = [list_ts[i].name for i in range(N)]
    dist_matrix = pd.DataFrame(index=list_names, data=np.zeros((N, N)), columns=list_names)

    # Compute dtw distances
    for i in range(N):
        # Diagonal elements left untouched (null by definition)
        for j in range(i+1, N):
            dist_ij = distance_model(list_ts[i], list_ts[j], **kwargs)
            dist_matrix.iloc[i, j] = dist_ij
            # Use symmetry
            dist_matrix.iloc[j, i] = dist_ij

    # Return matrix
    if normalize:
        dist_matrix_min = dist_matrix.values.min()
        dist_matrix_max = dist_matrix.values.max()
        return 2 * (dist_matrix - dist_matrix_min) / (dist_matrix_max - dist_matrix_min) - 1.
    else:
        return dist_matrix


@typechecked
def support_dtw_distance_matrix_from_ts_multi_proc(info: (np.ndarray, list, list, int, Callable[[], float], list)
                                                   ) -> np.ndarray:
    """
    Support function for dtw_distance_matrix_from_ts_multi_proc().
    Performs the DTW distance matrix calculation for the time series of interest.
    The calculation is done by one processor, through all the rows that are requested
    by 'parts' and restricting the evaluations to the upper triangle part of the matrix.

    Arguments
    ---------
    info : np.ndarray, list, list, int, Callable, int, str
      All necessary variables for the evaluation:
      - block: the empty matrix to be filled.
      - list_ts: the time series to be used for that.
      - parts: the list of values defining the matrix segmentation.
      - k: the part to be computed in this block.
      - distance_model: model to compute the distance between time series.
      - **kwargs: extra arguments for the distance_model function.

    Returns
    -------
    np.ndarray
      Array corresponding to the segmented block of the DTW distance matrix.
    """

    # Unpacking the transferred quantities for a single processor calculation
    (block, list_ts, parts, k, distance_model, kwargs) = info

    # Compute dtw distances
    # Run over the block's rows
    for i in range(block.shape[0]):
        # Run over the block's columns,
        # avoiding some part of the matrix lower triangle
        for j in range(parts[k-1]+i, block.shape[1]):
            block[i,j] = distance_model(list_ts[parts[k-1]+i], list_ts[j], **kwargs)

    return np.array(block)


#@typechecked
def distance_matrix_from_ts_multi_proc(list_ts, #: list,
                                       n_proc=None, #: Optional[int] = None,
                                       distance_model=dtw_distance, #: Callable[[], float] = dtw_distance,
                                       normalize=False, #: bool = False,
                                       **kwargs #: Any
                                       ): # -> pd.DataFrame:
    """
    Computes the dtw distance between time series of a list. It uses multi-processing
    with n_proc processors in parallel.

    Parameters
    ----------
    list_ts : list
      List of time series.
    n_proc : int
      Number of processors to use.
      If None, uses the machine's number of processors minus 1.
    distance_model : function
      Model to compute the distance between time series.
    normalize : bool
      Option to normalize the matrix (values between -1 and 1).
    **kwargs :
      Extra arguments for the distance_model function.

    Returns
    -------
    pd.DataFrame
      DataFrame containing dtw-distances between time series.
    """

    # Checks
    N = len(list_ts)
    if N < 2:
        raise AssertionError("Argument list_ts must have at least 2 time series in it.")

    # Initializations
    list_names = [list_ts[i].name for i in range(N)]
    parts = np.ceil(np.linspace(0, N, min(n_proc, N)+1)).astype(int)
    if n_proc is None:
        n_proc = mp.cpu_count() - 1

    # Creating the jobs
    jobs = []
    for k in range(1, len(parts)):
        block = np.zeros(shape=(parts[k]-parts[k-1],N))
        jobs.append((block, list_ts, parts, k, distance_model, kwargs))

    # Running the jobs and combining results
    pool = mp.Pool(processes=n_proc)
    outputs = pool.map(support_dtw_distance_matrix_from_ts_multi_proc, jobs)
    is_started = False
    for output_block in outputs:
        if not is_started:
            matrix = output_block.copy()
            is_started = True
        else:
            matrix = np.concatenate((matrix, output_block))
    pool.close()
    pool.join()

    # Copying the upper triangle into the lower triangle
    for i in range(1, N):
        for j in range(0, i):
            matrix[i, j] = matrix[j, i]

    # Return matrix
    dtw_matrix = pd.DataFrame(index=list_names, data=matrix, columns=list_names)
    if normalize:
        dtw_matrix_min = dtw_matrix.values.min()
        dtw_matrix_max = dtw_matrix.values.max()
        return 2 * (dtw_matrix - dtw_matrix_min) / (dtw_matrix_max - dtw_matrix_min) - 1.
    else:
        return dtw_matrix



# KMEANS CLUSTERING

@typechecked
def kmeans_base_clustering(corr: Union[np.ndarray, pd.DataFrame],
                           names_features: list=None,
                           max_num_clusters: int=10,
                           **kwargs: Any
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
        kmeans_current = cluster.KMeans(n_clusters=i, **kwargs).fit(X)

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
                               **kwargs: Any
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
        raise TypeError("max_num_clusters must be integer.")
      
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
        

@typechecked
def convert_clusters_into_list_of_labels(list_ts: list, clusters: dict) -> list:
    """

    Parameters
    ----------
    list_ts : list of ts.TimeSeries
      List of time series used to compute labels.
    clusters : dict
      Dictionary containing the labels and time series names.

    Returns
    -------
    list of int
      List with the clustering labels.
    """

    # Get the names and labels as they were returned from clustering
    clustered_names = []
    clustered_labels = []
    for cluster in clusters.items():
        for name in cluster[1]:
            clustered_names.append(name)
            clustered_labels.append(cluster[0])

    # Reorganise them in the same way as it is in list_ts
    labels = []
    for tsi in list_ts:
        labels.append(clustered_labels[clustered_names.index(tsi.name)])

    return labels


@typechecked
def cluster_observation_matrix(X: pd.DataFrame,
                               n_clust_range: range,
                               model: cluster,
                               verbose: bool=True,
                               **kwargs: Any
                               ) -> (dict, dict):
    """
    Apply clustering for an arbitrary model as long as the model has an argument 'n_clusters'.

    Parameters
    ----------
    X : pd.DataFrame
      The Observation matrix on which the clustering is based.
    n_clust_range: range
      Range of integer values for the number of clusters to be tested.
    model : sklearn.cluster
      The clustering model to be used from sklearn.
    verbose : bool
      Verbose option.
    **kwargs :
      Arguments for the clustering model.

    Returns
    -------
    dict
      Labels corresponding to the different clusters.
    dict
      Quality corresponding to the different clusters.

    Notes
    -----
      To learn more about sklearn.cluster:
      https://scikit-learn.org/stable/modules/classes.html?highlight=cluster#module-sklearn.cluster
    """

    # Checks
    if min(n_clust_range) < 2:
        raise AssertionError("Argument n_clust_range must have values starting at values >= 2.")

    # Initialization
    save_labels = {}
    save_quality = {}
    n_clust_range_min = n_clust_range[0]
    n_clust_range_max = n_clust_range[-1]
    n_clust_range_step = int(n_clust_range[-1] - n_clust_range[-2])

    # Looping
    for k in n_clust_range:

        # Build clusters
        fitted_model = model(n_clusters=k, **kwargs).fit(X)
        save_labels[k] = fitted_model.labels_.tolist()

        # Compute scores
        silh = silhouette_samples(X, fitted_model.labels_)
        save_quality[k] = silh.mean() / silh.std()

    # Plot qualities
    if verbose:
        plt.xticks(ticks=n_clust_range)
        plt.plot(n_clust_range, list(save_quality.values()))

        # Make it cute
        plt.title("Normalized Silhouette Score")
        plt.xlabel("Number of clusters")
        plt.ylabel("Score")


    # Make bars containing the clusters composition
    m = len(save_labels)
    assert(m == len(n_clust_range))
    bars = np.zeros(shape=(m,n_clust_range_max))

    # Loop over max number of clusters
    for k in n_clust_range:

        # Count appearing values
        count_vals = []
        for j in range(n_clust_range_max):
            count_vals.append(int(save_labels[k].count(j)))

        # Distribute these values to build bars
        for i in range(n_clust_range_max):
            bars[(k-n_clust_range_min)//n_clust_range_step,i] = count_vals[i]


    # Plot clusters compositions with bar plot
    if verbose:
        plt.figure(figsize=(10,5))
        m = bars.shape[0]
        sum_bars = [0] * m

        for i in range(n_clust_range_max):
            if i>0:
                sum_bars += bars[:,i-1]
            plt.bar(n_clust_range, bars[:,i], width=0.8, bottom=sum_bars)

        # Make it cute
        plt.xticks(ticks=n_clust_range)
        plt.title("Composition of clusters")
        plt.xlabel("Number of clusters")
        plt.ylabel("Composition")

    # Return labels
    return save_labels, save_quality


@typechecked
def reorganize_observation_matrix(X: pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    Reorganize a square observation matrix from labels obtained from clustering.

    Parameters
    ----------
    X : pd.DataFrame
      Observation matrix to reorganize.
    labels : list of int
      Labels to use to reorganize the matrix.

    Returns
    -------
    pd.DataFrame
      Reorganized observation matrix.
    """

    # Checks
    if X.shape[0] != X.shape[1]:
        raise AssertionError("The observation matrix must be a square matrix.")
    if X.shape[0] != len(labels):
        raise AssertionError("Argument labels must have the same dimension as the observation matrix side.")

    # Reoganize X according to this clustering
    new_idx = np.argsort(labels)
    clustered_X = X.iloc[new_idx].iloc[:, new_idx]

    return clustered_X


@typechecked
def print_clusters_content(list_ts: list, labels: list) -> None:
    """
    Print clusters content and cluster times series for a set of labels.

    Parameters
    ----------
    list_ts : list of ts.TimeSeries
      List of time series used to compute labels.
    labels : list of int
      Labels to apply to the time series.

    Returns
    -------
    None
      None
    """

    # Checks
    if len(list_ts) != len(labels):
        raise AssertionError("Arguments list_ts and labels must have same length.")

    # Get the clusters for this clustering
    list_ts_names = [tsi.name for tsi in list_ts]
    # print(list_ts_names)
    masks = [(labels==i) for i in np.unique(labels)]
    clusters = {}
    for i in range(len(masks)):
        clusters[i] = [list_ts_names[k] for k in range(len(list_ts_names)) if masks[i][k]]

    # Group time series according to clusters and print them
    for label in np.unique(labels):
        # Print cluster content names
        print("Composition of cluster " + str(label))
        print(clusters[label])
        # Plot cluster content time series
        cluster_idx = np.argsort(labels)[np.sort(labels)==label]
        cluster = [list_ts[idx] for idx in cluster_idx]
        print()
        print("Plotting cluster " + str(label))
        ts.multi_plot(cluster, title="Content of cluster"+str(label))

    return None


@typechecked
def show_clustering_results(list_ts: list, labels: list, expected_truth: list) -> pd.DataFrame:
    """
    Compute the data frame showing the cluster content from a clustering.

    Parameters
    ----------
    list_ts : list of ts.TimeSeries
      List of time series used to compute labels.
    labels : list of int
      Labels to apply to the time series.
    expected_truth : list of str
      Name of expected clusters true content.

    Returns
    -------
    pd.DataFrame
      Data Frame showing the content of clusters.
    """

    # Checks
    if len(list_ts) != len(labels):
        raise AssertionError("Arguments list_ts and labels must have same length.")

    # Initializations
    k = len(np.unique(labels))
    N = len(expected_truth)

    # Get the clusters for this clustering
    list_ts_names = [tsi.name for tsi in list_ts]
    masks = [(labels == i) for i in np.unique(labels)]
    clusters = {}
    for i in range(len(masks)):
        clusters[i] = [list_ts_names[k] for k in range(len(list_ts_names)) if masks[i][k]]

    # Analyse clusters content
    resu = np.zeros(shape=(N, k))
    reverse_dict = dict(zip(expected_truth, range(len(expected_truth))))
    for i in clusters.keys():
        for name in clusters[i]:
            resu[i, reverse_dict[name[:2]]] += 1

    # Build a DataFrame
    resu_df = pd.DataFrame(index=["Cluster " + str(i) for i in range(k)],
                           data=resu.astype(int),
                           columns=expected_truth)

    return resu_df



# FEATURE IMPORTANCE

@typechecked
def generate_random_classification(n_features: int,
                                   n_informative: int,
                                   n_redundant: int,
                                   n_samples: int,
                                   random_state: int=0,
                                   sigma_std: float=0.
                                   ) -> (pd.DataFrame, pd.Series):
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
    for arg in [('n_features', n_features), ('n_informative', n_informative), ('n_redundant', n_redundant),
                ('n_samples', n_samples), ('random_state', random_state)]:
        if not isinstance(arg[1], int):
            raise TypeError(arg[0] + " must be integer.")
    if not isinstance(arg[1], float) and not isinstance(arg[1], int):
        raise TypeError("sigma_std must be float.")
    
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


@typechecked
def feature_importance_pvalues(fit: statsmodels.discrete.discrete_model.BinaryResultsWrapper,
                               plot: bool=False,
                               figsize: (float,float)=(10,10)
                               ) -> pd.DataFrame:
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


@typechecked
def feature_importance_mdi(classifier: BaggingClassifier,
                           X: pd.DataFrame,
                           y: pd.Series,
                           plot: bool=False,
                           figsize: (float,float)=(10,10)
                           ) -> pd.DataFrame:
    """
    Feature importance based on in-sample Mean-Decrease Impurity (MDI).
    
    Arguments
    ---------
    classifier : sklearn.ensemble._bagging.BaggingClassifier
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


@typechecked
def feature_importance_mda(classifier: BaggingClassifier,
                           X: pd.DataFrame,
                           y: pd.Series,
                           n_splits: int=10,
                           plot: bool=False,
                           figsize: (float,float)=(10,10)
                           ) -> pd.DataFrame:
    """
    Feature importance based out-of-sample Mean-Decrease Accuracy (MDA).
    
    Arguments
    ---------
    classifier : sklearn.ensemble._bagging.BaggingClassifier
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