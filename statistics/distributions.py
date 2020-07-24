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
        # Type of distribution
        self.type = None
        
        # moments
        self.mean = None
        self.variance = None
        self.std = None
        self.skewness = None
        self.kurtosis = None
        
        # quantiles
        self.median = None
        
        # others
        self.mode = None
        
        # name (or nickname)
        self.name = name
    
    
    # Set functions
    def set_moments(self, mean, variance, skewness, kurtosis):
        """
        Sets the most commonly used moments.
        """
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(variance)
        self.skewness = skewness
        self.kurtosis = kurtosis
        
    def set_quantiles(self, median):
        """
        Sets the most commonly used quantiles.
        """
        self.median = median
        
    def set_mode(self, mode):
        """
        Sets the mode.
        """
        self.mode = mode
        
    def set_name(self, name):
        """
        Sets the name (or nickname).
        """
        self.name = name
        
    
    # Get functions
    def get_name(self):
        """
        Class method that returns the name given to the distribution.
        """
        return self.name
        
    def get_info(self):
        """
        Prints the most relevant information about the distribution.
        """
        print("Name: \t\t", self.name)
        print("Type: \t\t", self.type)
        print("Mean: \t\t", self.mean)
        print("Variance: \t", self.variance)
        print("Skewness: \t", self.skewness)
        print("Kurtosis: \t", self.kurtosis)
        print("Median: \t", self.median)
        print("Mode: \t\t", self.mode)

        
        
        
        
class Normal(distribution):
    """
    Class implementing the normal distribution, inheriting from class distribution.
    """
    
    def __init__(self, mu=0., sigma=1., name=""):
        """
        Initilialization function.
        """
        self.type = 'Normal'
        self.mu = mu
        self.sigma = sigma
        self.set_moments(mean=mu, variance=sigma*sigma, skewness=0., kurtosis=3.)
        self.set_quantiles(median=mu)
        self.set_mode(mode=mu)
        self.set_name(name)
        

    def PDF(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Normal distribution.
        """
        PDF = np.exp(-(np.array(x)-self.mu) * (np.array(x)-self.mu) / (2 * self.sigma * self.sigma)) / (self.sigma * np.sqrt(2 * np.pi))
        return PDF
