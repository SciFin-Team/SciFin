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
    Compute eigen value and eigen vector from a Hermitian matrix.
    
    Arguments
    ---------
    matrix : numpy.array or list of lists
      Matrix to extract eigen values and eigen vectors from.
      
    Returns
    -------
    numpy.array, numpy.array
      Array of eigen values, array of eigen vectors.
      
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
    
    
    # Compute eigen values and vectors
    e_val, e_vec = np.linalg.eigh(matrix)
    
    # Sort by size
    indices = e_val.argsort()[::-1]
    e_val = e_val[indices]
    e_vec = e_vec[:, indices]
    
    # Create diagonal matrix with eigenvalues
    e_val = np.diagflat(e_val)
    
    return e_val, e_vec


