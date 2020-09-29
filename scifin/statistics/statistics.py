# Created on 2020/7/24

# This module is for simple statistics.

# Standard library imports
from typing import Any, Union

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import minimize
import scipy.stats as ss
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.neighbors.kde import KernelDensity
from sklearn.utils import check_random_state
from typeguard import typechecked

# Local application imports
# /


#---------#---------#---------#---------#---------#---------#---------#---------#---------#


# COVARIANCE MATRICES

@typechecked
def random_covariance_matrix(n_features: int, n_obs: int) -> np.ndarray:
    """
    Generate a random matrix with dimensions n_features x n_features,
    simulating n_features x n_obs to generate the data.
    
    Parameters
    ----------
    n_features : int
      Dimension of the covariance matrix.
    n_obs : int
      Number of facts to generate data.
      
    Returns
    -------
    numpy.array
      Covariance matrix.
    """

    # Generate random numbers
    w = np.random.normal(size=(n_features, n_obs))
    
    # Random covariance matrix, not full rank
    cov = np.dot(w, w.T)
    
    # Full rank covariance matrix
    cov += np.diag(np.random.uniform(size=n_features))
    
    return cov


@typechecked
def covariance_to_correlation(cov: Union[list, np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Derive the correlation matrix from the covariance matrix.
    
    Arguments
    ---------
    cov : list of lists, np.array, pd.DataFrame
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


@typechecked
def correlation_to_covariance(corr: Union[list, np.ndarray], std: Union[list, np.ndarray]) -> np.ndarray:
    """
    Derive the covariance matrix from the correlation matrix.

    Arguments
    ---------
    corr : numpy.array or list of lists
      Correlation matrix.
    std : numpy.array or list
      Standard deviations.

    Returns
    -------
    numpy.array
      Covariance matrix.
    """
    
    cov = corr * np.outer(std, std)
    
    return cov


@typechecked
def covariance_from_ts(list_ts: list, **kwargs: Any) -> pd.DataFrame:
    """
    Compute the covariance matrix of a list of time series.
    
    Arguments
    ---------
    list_ts : list of TimeSeries
      List of time series we want to extract the covariance from.
    **kwargs :
      Arguments for pandas.DataFrame.cov().
    
    Returns
    -------
    pd.DataFrame
      Covariance matrix between the time series.
    
    Notes
    -----
      Makes use of pandas.DataFrame.cov(), more information here:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cov.html
    """

    # Initializations
    N = len(list_ts)
    idx = list_ts[0].data.index
    df = pd.DataFrame(index=idx, data=None)

    # Loop over values
    for i in range(N):

        # Indexes must be identical
        if (list_ts[i].data.index != idx).all():
            raise AssertionError("Time series must have same index values.")

        # Add time series data to the DataFrame
        df[list_ts[i].name] = list_ts[i].data

    # Compute the covariance matrix
    cov = df.cov(**kwargs)
    
    return cov


@typechecked
def denoise_covariance(cov: np.ndarray,
                       n_obs: int,
                       sigma_ini: float,
                       bwidth: float,
                       kernel: str='gaussian',
                       n_pts: int=1000):
    """
    Denoise a covariance matrix.
    
    Arguments
    ---------
    cov : numpy.array
      Covariance matrix to denoise.
    n_obs : int
      Number of facts to generate data.
    sigma_ini : float
      Initial value for standard deviation of observations.
    bwidth : float
      Bandwidth values.
    kernel : KernelDensity
      Kernel used to fit observations.
    n_pts : int
      Number of points to sample the PDF.

    Returns
    -------
    np.array
      Denoised covariance matrix.

    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Initialization
    n_features = cov.shape[0]
    
    # Compute correlation matrix
    corr = covariance_to_correlation(cov)
    
    # Compute eigenvalues and eigenvectors
    e_val, e_vec = eigen_value_vector(corr)
    
    # Find max eigenvalue from fit
    e_max, sigma, n_obs_new = marcenko_pastur_fit_params(n_features=n_features,
                                                         n_obs=n_obs,
                                                         sigma_ini=sigma_ini,
                                                         e_val=e_val,
                                                         bwidth=bwidth,
                                                         kernel=kernel,
                                                         n_pts=n_pts)

    # Remove noise from correlation matrix
    # by fixing random eigenvalues.
    eval_ = np.diag(e_val).copy()
    eval_[n_obs_new:] = eval_[n_obs_new:].sum() / float(eval_.shape[0] - n_obs_new)
    eval_ = np.diag(eval_)
    
    # Form new covariance matrix
    cov1 = np.dot(e_vec, eval_).dot(e_vec.T)
    
    return cov1


@typechecked
def get_subcovariance(n_features: int,
                      n_obs: int,
                      sigma: float,
                      random_state: Union[int, float, np.random.mtrand.RandomState]=None
                      ) -> np.ndarray:
    """
    Generate sub-covariance matrix with dimensions n_features x n_features,
    simulating n_features x n_obs to generate the data.
    
    Parameters
    ----------
    n_features : int
      Dimension of the covariance matrix.
    n_obs : int
      Number of facts to generate data.
    sigma : float
      Standard deviation of the noise.
    random_state : int, float, np.random.mtrand.RandomState
      Random State.
      
    Returns
    -------
    numpy.array
      Sub-covariance matrix.
      
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Initializations
    rng = check_random_state(random_state)
    
    if n_features==1:
        return np.ones((1,1))
    
    ar0 = rng.normal(size=(n_obs,1))
    ar0 = np.repeat(ar0, n_features, axis=1)
    ar0 += rng.normal(scale=sigma, size=ar0.shape)
    cov = np.cov(ar0, rowvar=False)
    
    return cov


@typechecked
def random_block_covariance(n_features: int,
                            n_blocks: int,
                            min_block_size: int=1,
                            sigma: float=1.,
                            random_state: Union[int, float, np.random.mtrand.RandomState]=None
                            ) -> np.ndarray:
    """
    Generate a block random covariance matrix.
    
    Parameters
    ----------
    n_features : int
      Dimension of the covariance matrix.
    n_blocks : int
      Number of blocks to generate.
    min_block_size : int
      Minimum block size.
    sigma : float
      Standard deviation of the noise.
    random_state : int, float, np.random.mtrand.RandomState
      Random State.
      
    Returns
    -------
    numpy.array
      Block random covariance matrix.
      
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Initializations
    rng = check_random_state(random_state)
    
    # Dimensionality check
    try:
        assert(n_features - (min_block_size-1) * n_blocks > 0)
    except AssertionError:
        raise AssertionError(f"n_features - (min_block_size-1) * n_blocks = {n_features - (min_block_size-1) * n_blocks} !")
        
    # Generate parts
    parts = rng.choice(range(1, n_features - (min_block_size-1) * n_blocks), n_blocks-1, replace=False)
    parts.sort()
    parts = np.append(parts, n_features - (min_block_size-1) * n_blocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + min_block_size
    
    # Generate covariance
    cov = None
    for n_features_ in parts:
        n_obs_ = int(max(n_features*(n_features_ + 1)/2., 100))
        cov_ = get_subcovariance(int(n_features_), n_obs_, sigma, random_state=rng)
        if cov is None:
            cov = cov_.copy()
        else:
            cov = block_diag(cov, cov_)
            
    return cov


@typechecked
def random_block_correlation(n_features: int,
                             n_blocks: int,
                             min_block_size: int=1,
                             sigma_base: float=1.,
                             sigma_noise: float=0.5,
                             random_state: Union[int, float, np.random.mtrand.RandomState]=None
                             ) -> pd.DataFrame:
    """
    Form block random correlation matrix of dimension n_features x n_features with n_blocks blocks.
    Minimum block size and noise levels can be specified.
    
    Parameters
    ----------
    n_features : int
      Dimension of the covariance matrix.
    n_blocks : int
      Number of blocks to generate.
    min_block_size : int
      Minimum block size.
    sigma_base : float
      Standard deviation of the noise in base blocks.
    sigma_noise : float
      Standard deviation for the noise added to each matrix element.
    random_state : int, float, np.random.mtrand.RandomState
      Random State.
      
    Returns
    -------
    numpy.array
      Block random correlation matrix.
      
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """

    # Initializations
    rng = check_random_state(random_state)

    # Initial random block covariance matrix
    cov_base = random_block_covariance(n_features, n_blocks, min_block_size=min_block_size,
                                       sigma=sigma_base, random_state=rng)
    
    # Add noise
    cov_noise = random_block_covariance(n_features, 1, min_block_size=min_block_size,
                                        sigma=sigma_noise, random_state=rng)
    cov = cov_base + cov_noise
    
    # Generate correlation matrix
    corr = covariance_to_correlation(cov)
    corr = pd.DataFrame(corr)
    
    return corr



# EIGENVALUES AND EIGENVECTORS

@typechecked
def eigen_value_vector(matrix: Union[list, np.ndarray]) -> (np.ndarray, np.ndarray):
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
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
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


@typechecked
def marcenko_pastur_pdf(n_features: int,
                        n_obs: int,
                        sigma: float,
                        n_pts: int=100,
                        verbose: bool=False
                        ) -> pd.Series:
    """
    Implements the Marcenko-Pastur PDF from the characteristics of
    an observation matrix representing IID random observations with
    zero mean and standard deviation sigma.
    
    Arguments
    ---------
    n_features : int
      Number of features.
    n_obs : int
      Number of observations (in time).
    sigma : float
      Standard deviation of observations.
    n_pts : int
      Number of points to sample the PDF.
    verbose : bool
      Verbose option.
      
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
        
    # Initializations
    ratio = n_obs / n_features
    
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

@typechecked
def marcenko_pastur_loss(sigma: Union[float, np.ndarray],
                         n_features: int,
                         n_obs: int,
                         e_val: np.ndarray,
                         bwidth: float,
                         kernel: str='gaussian',
                         n_pts: int=1000
                         ) -> float:
    """
    Return the loss (sum of squared errors) from the Marcenko-Pastur distribution.
    
    Arguments
    ---------
    sigma : float
      Standard deviation of observations.
    n_features : int
      Number of features.
    n_obs : int
      Number of observations (in time).
    e_val : np.array
      Diagonal matrix of eigenvalues.
    bwidth : float
      Bandwidth values.
    kernel : KernelDensity
      Kernel used to fit observations.
    n_pts : int
      Number of points to sample the PDF.

    Returns
    -------
    float
      Loss value.

    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """

    # Compute Theoretical PDF
    pdf0 = marcenko_pastur_pdf(n_features, n_obs, sigma[0], n_pts)

    # Compute Empirical PDF
    # Fit kernel to a series of observations
    if len(e_val.shape)==1:
        e_val = e_val.reshape(-1,1)
    kde = KernelDensity(kernel=kernel, bandwidth=bwidth).fit(e_val)
    # Create index
    x = pdf0.index.values
    if len(x.shape)==1:
        x = x.reshape(-1,1)
    # Derive the probability of observations
    log_density = kde.score_samples(x)
    pdf1 = pd.Series(np.exp(log_density), index=x.flatten())

    # Return loss
    loss = np.sum((pdf1-pdf0)**2)

    return loss


@typechecked
def marcenko_pastur_fit_params(n_features: int,
                               n_obs: int,
                               sigma_ini: float,
                               e_val: np.ndarray,
                               bwidth: float,
                               kernel: str='gaussian',
                               n_pts: int=1000
                               ) -> (np.ndarray, float, int):
    """
    Find max random e_val by fitting the Marcenko-Pastur distribution.

    Arguments
    ---------
    n_features : int
      Number of features.
    n_obs : int
      Number of observations (in time).
    sigma_ini : float
      Initial value for standard deviation of observations.
    e_val : np.array
      Matrix of eigenvalues.
    bwidth : float
      Bandwidth values.
    kernel : KernelDensity
      Kernel used to fit observations.
    n_pts : int
      Number of points to sample the PDF.
    
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Checks
    assert(n_features == len(e_val))
    
    # Initializations
    ratio = n_obs/n_features
    
    # Minimize loss
    loss_function = (lambda *x: marcenko_pastur_loss(*x))
    out = minimize(loss_function,
                   x0=sigma_ini,
                   args=(n_features, n_obs, np.diag(e_val), bwidth, kernel, n_pts),
                   bounds=((1E-5, 1-1E-5),))
    
    if out['success']:
        sigma = out['x'][0]
    else:
        sigma = 1
        
    # Rescaling max expected eigenvalue
    e_max = sigma**2 * (1 + ratio**.5)**2
    
    # Compute n_obs
    n_obs = e_val.shape[0] - np.diag(e_val)[::-1].searchsorted(e_max)
    
    return e_max, sigma, n_obs



# SIMILARITY / DISSIMILARITY MEASURES

@typechecked
def distance_from_vectors(X: Union[list, np.ndarray], Y: Union[list, np.ndarray]) -> float:
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
    

@typechecked
def pearson_correlation(X: Union[list, np.ndarray], Y: Union[list, np.ndarray]) -> float:
    """
    Implements the Pearson's correlation between two random vectors X and Y.
    
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


@typechecked
def distance_from_pearson(X: Union[list, np.ndarray], Y: Union[list, np.ndarray]) -> float:
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


@typechecked
def distance_from_abs_pearson(X: Union[list, np.ndarray], Y: Union[list, np.ndarray]) -> float:
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


@typechecked
def entropy_info(X: Union[list, np.ndarray],
                 Y: Union[list, np.ndarray],
                 bins: int,
                 returns: str=None,
                 verbose: bool=False
                 ) -> Union[float, None]:
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
      Possible choices: ["joint", "marginal_X", "marginal_Y", "mutual_info", "mutual_info_norm", \
      "joint_entropy", "conditional_entropy_X", "conditional_entropy_Y", "variation_info", "variation_info_norm"]
    verbose : bool
      Verbose option.
    
    Returns
    -------
    None or float
      None if argument 'returns' is None. Otherwise returns the specified quantity.
      
    Notes
    -----
      Function adapted from "Machine Learning for Asset Managers",
      Marcos López de Prado (2020).
    """
    
    # Checks
    if len(X) != len(Y):
        raise AssertionError("Vectors X and Y must have same length.")
    if returns not in [None, "joint", "marginal_X", "marginal_Y", "mutual_info", "mutual_info_norm", \
      "joint_entropy", "conditional_entropy_X", "conditional_entropy_Y", "variation_info", "variation_info_norm"]:
        raise AssertionError("Argument 'returns' choice not correct.")
    
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
        return hX_Y
    hY_X = hXY - hX
    if (verbose==False) and (returns=="conditional_entropy_Y"):
        return hY_X
    
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
        print("Conditional entropy of Y given X: \t", hY_X)
        print("Variation of information: \t \t", vXY)
        print("Normalized variation of information: \t", vXYn)
        
    return None



