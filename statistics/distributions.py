# Created on 2020/7/24

# This module is for probability distributions.

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv


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
        self.support = None
        
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
        self.MAD = None
        
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
        
    def set_MAD(self, MAD):
        """
        Sets the Mean Absolute Deviation (MAD)
        """
        self.MAD = MAD
        
    def set_entropy(self, entropy):
        """
        Sets the entropy of the distribution.
        """
        self.entropy = entropy
        
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
        print("MAD: \t\t", self.MAD)
        print("Entropy: \t", self.entropy)
        
        
        
        
class Normal(distribution):
    """
    Class implementing the normal distribution, inheriting from class distribution.
    """
    
    def __init__(self, mu=0., sigma=1., name=""):
        """
        Initilialization function.
        """
        assert(sigma>0)
        # Type of distribution
        self.type = 'Normal'
        self.support = 'R'
        # moments
        self.mu = mu
        self.sigma = sigma
        self.set_moments(mean = mu, variance = sigma*sigma, skewness = 0., kurtosis = 3.)
        # quantiles
        self.set_quantiles(median = mu)
        # others
        self.set_mode(mode = mu)
        self.set_MAD(MAD = sigma*np.sqrt(2/np.pi))
        self.set_entropy(entropy = (1/2) * np.log(2*np.pi*np.e*sigma*sigma))
        # name (or nickname)
        self.set_name(name)
        
    def PDF(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Normal distribution.
        """
        return np.exp(-(np.array(x)-self.mu) * (np.array(x)-self.mu) / (2 * self.sigma * self.sigma)) / (self.sigma * np.sqrt(2 * np.pi))

    
    def CDF(self, x):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the Normal distribution.
        """
        return [(1/2) * (1 + erf((x_el - self.mu) / (self.sigma * np.sqrt(2)))) for x_el in x]
    
    
    def quantile(self, p):
        """
        Method returning the quantile associated to a certain probability.
        """
        assert(p>0 and p<1)
        return self.mu + self.sigma * np.sqrt(2) * erfinv(2*p-1)
    
    
    
    
    