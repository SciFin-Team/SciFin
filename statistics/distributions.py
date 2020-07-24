# Created on 2020/7/24

# This module is for probability distributions.

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv


support = ['R', 'R+', 'N']


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
        
    def set_median(self, median):
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
        self.set_median(median = mu)
        
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
    
    
    
class Poisson(distribution):
    """
    Class implementing the Poisson distribution, inheriting from class distribution.
    We use a value k_max to set the limit of summation for the entropy calculation.
    """
    
    def __init__(self, lmbda=0., k_max=1000, name=""):
        """
        Initilialization function.
        """
        assert(lmbda>=0.)
        assert(isinstance(k_max,int))
        assert(k_max>0)
        
        # Type of distribution
        self.type = 'Poisson'
        self.support = 'N'
        
        # moments
        self.lmbda = lmbda
        self.set_moments(mean = lmbda, variance = lmbda, skewness = 1./np.sqrt(lmbda), kurtosis = 3. + 1./np.sqrt(lmbda))
        
        # quantiles
        self.set_median(median = np.floor(lmbda + 1/3 - 0.02/lmbda))
        
        # others
        self.k_max = k_max
        self.set_mode(mode = np.floor(lmbda))
        self.set_entropy(entropy = self.entropy_Poisson(self.lmbda))
        
        # name (or nickname)
        self.set_name(name)
    
    
    def entropy_Poisson(self, lmbda):
        """
        Computes the entropy for the Poisson distribution.
        """
        tmp_sum = 0.
        for k in range(self.k_max):
            contrib = np.power(lmbda,k) * np.log(np.math.factorial(k)) / np.math.factorial(k)
            if contrib < 1.e-15:
                tmp_sum += contrib
                break
        if k==self.k_max:
            print("Careful. Sum probably did not converge.")
        return lmbda * (1-np.log(lmbda)) + np.exp(-lmbda) * tmp_sum
        
        
    def PMF(self, klist):
        """
        Method implementing the Probability Mass Function (PMF) for the Poisson distribution.
        """
        assert(len(klist)>0)
        
        for x in klist:
            assert(isinstance(x,int))
        pmf = [np.power(self.lmbda,x) * np.exp(-self.lmbda) / np.math.factorial(x) for x in klist]
        return pmf

    
    def CDF(self, klist):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the Poisson distribution.
        """
        # Checks
        assert(len(klist)>0)
        N = len(klist)
        for x in klist:
            assert(isinstance(x,int))
            
        # Check if k's are consecutive
        ks_consecutive = True
        for i in range(N-1):
            if (klist[i+1] != klist[i]+1):
                ks_consecutive = False
                break
        
        # Computing the list of elements to sum
        t = []
        if ks_consecutive==True:
            k_range = np.floor(klist)
            tmp_sum = 1.
            t.append(tmp_sum)
            for i in range(1,N,1):
                tmp_sum += np.power(self.lmbda,k_range[i]) / np.math.factorial(k_range[i])
                t.append(tmp_sum)
        else:
            for i in range(N):
                tmp_sum = [np.power(self.lmbda,i) / np.math.factorial(i) for i in range(np.floor(k_range[i]))].sum()
                t.append(tmp_sum)
        
        # Completing the calculation
        return np.exp(-self.lmbda) * np.array(t)
    
    
    