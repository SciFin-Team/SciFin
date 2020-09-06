# Created on 2020/7/24

# This module is for simple statistics.

# Standard library imports
# /

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as ss
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.neighbors.kde import KernelDensity

# Local application imports
# /



#---------#---------#---------#---------#---------#---------#---------#---------#---------#


# COVARIANCE MATRICES

def random_covariance_matrix(n_features, n_facts):
    """
    Generate a random matrix with dimensions n_features x n_features,
    simulating n_features x n_facts to generate the data.
    
    Parameters
    ----------
    n_features : int
      Dimension of the covariance matrix.
    n_facts : int
      Number of facts to generate data.
      
    Returns
    -------
    numpy.array
      Covariance matrix.
    """
    
    # Checks
    if isinstance(n_features, int) is False:
        raise AssertionError("Argument n_features for matrix dimension must be integer.")
    if isinstance(n_facts, int) is False:
        raise AssertionError("Argument n_facts must be integer.")
    
    # Generate random numbers
    w = np.random.normal(size=(n_features, n_facts))
    
    # Random covariance matrix, not full rank
    cov = np.dot(w, w.T)
    
    # Full rank covariance matrix
    cov += np.diag(np.random.uniform(size=n_features))
    
    return cov


def covariance_to_correlation(cov):
    """
    Derive the correlation matrix from the covariance matrix.
    
    Arguments
    ---------
    cov : numpy.array or list of lists
      Covariance matrix.
      
    Returns
    -------
    numpy.array
      Correlation matrix.
    """
    
    # Convert list of lists to numpy.array (if needed)
    if isinstance(cov, list):
        cov = np.array(cov)

    # Checks
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be a square matrix.")
    
    # Compute correlation matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    
    # Deal with potential numerical errors
    corr[corr < -1] = -1
    corr[corr > 1] = 1
    
    return corr


def denoise_cov(cov0, q, b_width):
    
    corr0 = cov_to_corr(cov0)
    e_val0, e_vec0 = get_pca(corr0)
    e_max0, var0 = find_max_eval(np.diag(e_val0), q, b_width)
    n_facts0 = e_val0.shape[0] - np.diag(e_val0)[::-1].searchsorted(e_max0)
    corr1 = denoised_corr(e_val0, e_vec0, n_facts0)
    cov1 = corr_to_cov(corr1, np.diag(cov0)**.5)
    
    return cov1


def covariance_from_ts(list_ts, **kwargs):
    """
    Compute the covariance matrix of a list of time series.
    
    Arguments
    ---------
    list_ts : list of TimeSeries
      List of time series we want to extract the covariance from.
    **kwargs:
      Arguments for pandas.DataFrame.cov().
    
    Returns
    -------
    numpy.array
      Covariance matrix between the time series.
    
    Notes
    -----
      Makes use of pandas.DataFrame.cov(), more information here:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cov.html
    
    """
    
    # Checks
    N = len(list_ts)
    T = list_ts[0].nvalues
    # Indexes must be identical
    idx = list_ts[0].data.index
    for i in range(1,N,1):
        if (list_ts[i].data.index != idx).all():
            raise AssertionError("Time series must have same index values.")
    # Names must be different
    set_names = set()
    for i in range(N):
        set_names.add(list_ts[i].name)
    if len(set_names) != N:
        raise AssertionError("Names of time series must be different")
            
    # Make a data frame
    df = pd.DataFrame(index=idx, data=None)
    for i in range(N):
        df[list_ts[i].name] = list_ts[i].data
    
    # Compute the covariance matrix
    cov = df.cov(**kwargs)
    
    return np.array(cov)



# EIGENVALUES AND EIGENVECTORS

def eigen_value_vector(matrix):
    """
    Compute eigenvalue and eigenvector from a Hermitian matrix.
    
    Arguments
    ---------
    matrix : numpy.array or list of lists
      Matrix to extract eigenvalues and eigenvectors from.
      
    Returns
    -------
    numpy.array, numpy.array
      Array of eigenvalues, array of eigenvectors.
      
    Notes
    -----
      Function adapted from "Advances in Financial Machine Learning",
      Marcos López de Prado (2018).
    """
    
    # Convert list of lists to numpy.array (if needed)
    if isinstance(matrix, list):
        matrix = np.array(matrix)
        
    # Checks
    # Squared matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be a square matrix.")
    # Hermitian
    if (matrix.T == np.conj(matrix)).flatten().all() == False:
        raise ValueError("Input matrix must be Hermitian.")
    
    # Compute eigenvalues and eigenvectors
    e_val, e_vec = np.linalg.eigh(matrix)
    
    # Sort by size
    indices = e_val.argsort()[::-1]
    e_val = e_val[indices]
    e_vec = e_vec[:, indices]
    
    # Create diagonal matrix with eigenvalues
    e_val = np.diagflat(e_val)
    
    return e_val, e_vec


def marcenko_pastur_pdf(n_features, n_facts, sigma, n_pts=100, verbose=False):
    """
    Implements the Marcenko-Pastur PDF from the characteristics of
    an observation matrix representing IID random observations with
    zero mean and standard deviation sigma.
    
    Arguments
    ---------
    n_features : int
      Number of features.
    n_facts: int
      Number of observations (in time).
    sigma: float
      Standard deviation of observations.
    n_pts: int
      Number of points to sample the PDF.
    
    Notes
    -----
      Function adapted from "Advances in Financial Machine Learning",
      Marcos López de Prado (2018).
    """
    
    # Check
    if (isinstance(n_features, int)==False) or (isinstance(n_facts, int)==False):
        print(n_features, n_facts)
        raise AssertionError("n_features and n_facts must be integers.")
    if (isinstance(n_pts, int)==False):
        raise AssertionError("n_pts must be integer.")
        
    # Initializations
    ratio = n_facts / n_features
    
    # Max and min expected eigenvalues
    e_min = sigma**2 * (1 - ratio**.5)**2
    e_max = sigma**2 * (1 + ratio**.5)**2

    e_val = np.linspace(e_min, e_max, n_pts)
    pdf = ((e_max-e_val)*(e_val-e_min))**.5 / (2 * ratio * np.pi * sigma**2 * e_val)

    # Display information
    if verbose:
        print("Noise eigenvalues in range [" + str(e_min) + " , " + str(e_max) + "].")
        
    # Build series
    pdf = pd.Series(data=pdf.flatten(), index=e_val.flatten())
    
    return pdf


def marcenko_pastur_loss(sigma, n_features, n_facts, e_val, bwidth, kernel='gaussian', n_pts=100):
    """
    Return the loss (sum of squared errors) from the Marcenko-Pastur distribution.
    
    Arguments
    ---------
    sigma: float
      Standard deviation of observations.
    n_features : int
      Number of features.
    n_facts: int
      Number of observations (in time).
    e_val: matrix
      Diagonal matrix of eigenvalues.
    bwidth: float
      Bandwidth values.
    kernel: KernelDensity
      Kernel used to fit observations.
    n_pts: int
      Number of points to sample the PDF.
    
    Notes
    -----
      Function adapted from "Advances in Financial Machine Learning",
      Marcos López de Prado (2018).
    """
    
    # Compute Theoretical PDF
    pdf0 = marcenko_pastur_pdf(n_features, n_facts, sigma, n_pts)
    
    # Compute Empirical PDF
    # - Fit kernel to a series of observations
    if len(e_val.shape)==1:
        obs = e_val.reshape(-1,1)
    kde = KernelDensity(kernel=kernel, bandwidth=bwidth).fit(obs)
    # - Create index
    x = pdf0.index.values
    if x is None:
        x = np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:
        x = x.reshape(-1,1)
    # - Derive the probability of observations
    log_density = kde.score_samples(x)
    pdf1 = pd.Series(np.exp(log_density), index=x.flatten())

    # Return loss
    loss = np.sum((pdf1-pdf0)**2)
    return loss


def marcenko_pastur_fit_params(n_features, n_facts, sigma_ini, e_val, bwidth, kernel='gaussian'):
    """
    Find max random e_val by fitting the Marcenko-Pastur distribution.
    
    Arguments
    ---------
    n_features : int
      Number of features.
    n_facts: int
      Number of observations (in time).
    sigma_ini: float
      Initial value for standard deviation of observations.
    e_val: matrix
      Matrix of eigenvalues.
    bwidth: float
      Bandwidth values.
    kernel: KernelDensity
      Kernel used to fit observations.
    
    Notes
    -----
      Function adapted from "Advances in Financial Machine Learning",
      Marcos López de Prado (2018).
    """
    
    # Checks
    assert(n_features==len(e_val))
    
    # Initializations
    ratio = n_facts/n_features
    
    # Minimize loss
    out = minimize(lambda *x: marcenko_pastur_loss(*x), x0=sigma_ini, args=(n_features, n_facts, np.diag(e_val), bwidth), bounds=((1E-5, 1-1E-5),))
    
    if out['success']:
        sigma = out['x'][0]
    else:
        sigma = 1
        
    # Rescaling max expected eigenvalue
    e_max = sigma**2 * (1 + ratio**.5)**2
    
    # Compute n_facts
    n_facts = e_val.shape[0] - np.diag(e_val)[::-1].searchsorted(e_max)
    
    return e_max, sigma, n_facts




# SIMILARITY / DISSIMILARITY MEASURES

def distance_from_vectors(X, Y):
    """
    Implements the Euclidean distance between two random vectors X and Y.
    
    Arguments
    ---------
    X : numpy.array or list
      Random vector X.
    Y : numpy.array or list
      Random vector Y.
      
    Returns
    -------
    float
      Eucliden distance between vectors X and Y.
    """

    # Checks
    if len(X) != len(Y):
        raise AssertionError("Vectors X and Y must have same length.")
    
    # Initializations
    X = np.array(X)
    Y = np.array(Y)
    
    return np.sqrt(np.power(X-Y, 2).sum())
    
    
def pearson_correlation(X, Y):
    """
    Implements the Pearson's correlation betwenn two random vectors X and Y.
    
    Arguments
    ---------
    X : numpy.array or list
      Random vector X.
    Y : numpy.array or list
      Random vector Y.
      
    Returns
    -------
    float
      Pearson correlation coefficient between vectors X and Y.
    """
    
    # Checks
    if len(X) != len(Y):
        raise AssertionError("Vectors X and Y must have same length.")
    
    # Initializations
    N = len(X)
    X = np.array(X)
    Y = np.array(Y)
    X = X - X.mean()
    Y = Y - Y.mean()

    return np.dot(X,Y) / N / np.std(X) / np.std(Y)


def distance_from_pearson(X, Y):
    """
    Implements the Euclidean distance between two random vectors X and Y
    from Pearson's correlation.
    
    Arguments
    ---------
    X : numpy.array or list
      Random vector X.
    Y : numpy.array or list
      Random vector Y.
      
    Returns
    -------
    float
      Eucliden distance from Pearson coefficient, between vectors X and Y.
    """
    
    # Checks
    if len(X) != len(Y):
        raise AssertionError("Vectors X and Y must have same length.")
    
    return np.sqrt(0.5 * (1 - pearson_correlation(X,Y)))


def distance_from_abs_pearson(X, Y):
    """
    Implements the Euclidean distance between two random vectors X and Y
    from Pearson's correlation absolute value.
    
    Arguments
    ---------
    X : numpy.array or list
      Random vector X.
    Y : numpy.array or list
      Random vector Y.
      
    Returns
    -------
    float
      Eucliden distance from Pearson absolute coefficient, between vectors X and Y.
    """
    
    # Checks
    if len(X) != len(Y):
        raise AssertionError("Vectors X and Y must have same length.")
    
    return np.sqrt(1 - np.abs(pearson_correlation(X,Y)))


def entropy_info(X, Y, bins, returns=None, verbose=False):
    """
    Display entropy information between two random vectors X and Y.
    Or returns any quantity that is wished for.
    X and Y are assumed to be IID Gaussian random vectors.
    
    Arguments
    ---------
    X : numpy.array or list
      Random vector X.
    Y : numpy.array or list
      Random vector Y.
    bins : int
      Number of bins.
    returns : str
      Name of quantity to return.
    verbose : bool
      Verbose option.
    
    Returns
    -------
    None
      None
      
    Notes
    -----
      Function adapted from "Advances in Financial Machine Learning",
      Marcos López de Prado (2018).
    """
    
    # Checks
    if len(X) != len(Y):
        raise AssertionError("Vectors X and Y must have same length.")
    if not isinstance(bins, int):
        raise AssertionError("Value of bins must be integer.")
    
    # Initializations
    X = np.array(X)
    Y = np.array(Y)
    
    # Joint distribution
    cXY = np.histogram2d(X,Y,bins=bins)
    if (verbose==False) and (returns=="joint"):
        return cXY
    
    # Entropies of marginal distributions
    hX = ss.entropy(np.histogram(X, bins=bins)[0])
    if (verbose==False) and (returns=="marginal_X"):
        return hX
    hY = ss.entropy(np.histogram(Y, bins=bins)[0])
    if (verbose==False) and (returns=="marginal_Y"):
        return hY
    
    # Mutual information
    iXY = mutual_info_score(None, None, contingency=cXY[0])
    if (verbose==False) and (returns=="mutual_info"):
        return iXY
    iXYn = iXY/min(hX, hY)
    if (verbose==False) and (returns=="mutual_info_norm"):
        return iXYn
    
    # Joint entropy
    hXY = hX + hY - iXY
    if (verbose==False) and (returns=="joint_entropy"):
        return hXY
    
    # Conditional entropies
    hX_Y = hXY - hY
    if (verbose==False) and (returns=="conditional_entropy_X"):
        return hXY
    hX_Y = hXY - hY
    if (verbose==False) and (returns=="conditional_entropy_Y"):
        return hXY
    
    # Variation of information
    vXY = hX + hY - 2*iXY
    if (verbose==False) and (returns=="variation_info"):
        return vXY
    vXYn = vXY / hXY
    if (verbose==False) and (returns=="variation_info_norm"):
        return vXYn
    
    # Display
    if verbose:
        plt.figure(figsize=(8,5))
        sns.heatmap(cXY[0], xticklabels=np.around(cXY[1],2), yticklabels=np.around(cXY[2],2), cmap='viridis')
        plt.title("Joint distribution of X and Y.")
        plt.show()
        print("Entropy of marginal distribution in X: \t", hX)
        print("Entropy of marginal distribution in Y: \t", hY)
        print("Mutual information: \t \t \t", iXY)
        print("Normalized mutual information: \t \t", iXYn)
        print("Joint entropy of X and Y: \t \t", hXY)
        print("Conditional entropy of X given Y: \t", hX_Y)
        print("Conditional entropy of X given Y: \t", hX_Y)
        print("Variation of information: \t \t", vXY)
        print("Normalized variation of information: \t", vXYn)
        
    return None



