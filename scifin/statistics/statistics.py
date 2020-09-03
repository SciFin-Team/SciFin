# Created on 2020/7/24

# This module is for simple statistics.

# Standard library imports
# /

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

