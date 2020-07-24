# Created on 2020/7/24

# This module is for probability distributions.

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


support = ['R', 'R+']

class distribution:
    """
    General class for a distribution.
    """
    
    def __init__(self, name=""):
        """
        Initilialization function.
        """
        self.mean = None
        self.median = None
        self.var = None
        self.name = name
        
        
    def get_name(self):
        """
        Class method that returns the name given to the distribution.
        """
        return self.name
        
        
        

        
class Normal(distribution):
    """
    Class implementing the normal distribution, inheriting from class distribution.
    """
    
    def __init__(self, mu=0., sigma=1., name=""):
        """
        Initilialization function.
        """
        self.mu = mu
        self.sigma = sigma
        self.name = name
        

    def PDF(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Normal distribution.
        """
        PDF = np.exp(-(np.array(x)-self.mu) * (np.array(x)-self.mu) / (2 * self.sigma * self.sigma)) / (self.sigma * np.sqrt(2 * np.pi))
        return PDF
