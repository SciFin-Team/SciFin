# Created on 2020/7/24

# This module is for simple statistics.

# Standard library imports
# /

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Local application imports
# /



#---------#---------#---------#---------#---------#---------#---------#---------#---------#



def random_covariance_matrix(n, n_facts):
    """
    Generate a random matrix with dimensions n x n,
    simulating n x n_facts to generate the data.
    
    Parameters
    ----------
    n : int
      Dimension of the covariance matrix.
    n_facts : int
      Number of facts to generate data.
      
    Returns
    -------
    numpy.array
      Covariance matrix.
    """
    
    # Checks
    if isinstance(n, int) is False:
        raise AssertionError("Argument n for matrix dimension must be integer.")
    if isinstance(n_facts, int) is False:
        raise AssertionError("Argument n_facts must be integer.")
    
    # Generate random numbers
    w = np.random.normal(size=(n, n_facts))
    
    # Random covariance matrix, not full rank
    cov = np.dot(w, w.T)
    
    # Full rank covariance matrix
    cov += np.diag(np.random.uniform(size=n))
    
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


def marcenko_pastur_pdf(n_features, n_obs, sigma, n_pts=100):
    """
    Implements the Marcenko-Pastur PDF from the characteristics of
    an observation matrix representing IID random observations with
    zero mean and standard deviation sigma.
    
    Arguments
    ---------
    n_features : int
      Number of features.
    n_obs: int
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
    
    # Checks
    if (isinstance(n_features, int)==False) or (isinstance(n_obs, int)==False):
        raise AssertionError("n_features and n_obs must be integers.")
    if (isinstance(n_pts, int)==False):
        raise AssertionError("n_pts must be integer.")
        
    # Initializations
    ratio = n_features / n_obs
    
    # Max and min expected eigenvalues
    e_min = sigma**2 * (1 - ratio**.5)**2
    e_max = sigma**2 * (1 + ratio**.5)**2

    e_val = np.linspace(e_min, e_max, n_pts)
    pdf = ((e_max-e_val)*(e_val-e_min))**.5 / (2 * ratio * np.pi * sigma**2 * e_val)

    print("Noise eigenvalues in range [" + str(e_min) + " , " + str(e_max) + "].")
    pdf = pd.Series(data=pdf.flatten(), index=e_val.flatten())
    
    return pdf

