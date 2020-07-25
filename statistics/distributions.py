# Created on 2020/7/24

# This module is for probability distributions.

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv



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
        print("Entropy: \t", self.entropy)
        
        
        
        
# CONTINUOUS DISTRIBUTIONS
        
class Normal(distribution):
    """
    Class implementing the normal distribution of mean 'mu' and standard deviation 'sigma'.
    This class is inheriting from the class 'distribution'.
    """
    
    def __init__(self, mu=0., sigma=1., name=""):
        """
        Initilialization function.
        """
        assert(sigma>0)

        # Type of distribution
        self.type = 'Normal'
        self.support = 'R'
        
        # parameters
        self.mu = mu
        self.sigma = sigma
        
        # moments
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
    
    
    
    
# DISCRETE DISTRIBUTIONS
    
class Poisson(distribution):
    """
    Class implementing the Poisson distribution with expected rate of event occurence 'lambda'.
    This class is inheriting from the class 'distribution'.
    
    Note: We use a value k_max to set the limit of summation for the entropy calculation.
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
        
        # parameters
        self.lmbda = lmbda
        
        # moments
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
    
    
    
    
    
    
    
class Binomial(distribution):
    """
    Class implementing the binomial distribution of "successes" having probability of success 'p' in a sequence of 'n' independent experiments (number of trials). I also applies to drawing an element with probability p in a population of n elements, with replacement after draw. This class is inheriting from the class 'distribution'.
    
    Note: Default value for n is taken to be 1, hence corresponding to the Bernoulli distribution.
    Note 2: Computation of entropy is only an approximation valid at order O(1/n).
    """
    
    def __init__(self, n=1, p=0.5, name=""):
        """
        Initilialization function.
        """
        # Checks
        assert(n>=0)
        assert(isinstance(n,int))
        assert(p>=0 and p<=1)
        
        # Type of distribution
        self.type = 'Binomial'
        self.support = '{0,1,...,n}'
        
        # moments
        self.n = n
        self.p = p
        self.q = 1 - p
        self.set_moments(mean = n*p, variance = n*p*(1-p), skewness = ((1-p)-p)/np.sqrt(n*p*(1-p)), kurtosis = 3. + (1-6*p*(1-p))/(n*p*(1-p)))
        
        # quantiles
        self.set_median(median = self.median_Binomial(self.n, self.p))
        
        # others
        self.set_mode(mode = self.mode_Binomial(self.n, self.p))
        self.set_entropy(entropy = (1/2) * np.log2(2 * np.pi * np.e * n*p*(1-p)))
        
        # name (or nickname)
        self.set_name(name)
        

    def mode_Binomial(self, n, p):
        """
        Computes the mode of the Binomial distribution.
        """
        test_value = (n+1)*p
        if test_value == 0 or isinstance(test_value,int) == False:
            return np.floor(test_value)
        elif test_value in range(1,n,1):
            print("Binomial distribution for these values has two modes.")
            return (test_value, test_value-1)
        elif test_value == n+1:
            return n

        
    def median_Binomial(self, n, p):
        """
        Partially computes the median of the Binomial distribution.
        """
        test_value = n*p
        if test_value==int(test_value):
            return test_value
        else:
            print("Median has a value in interval [", np.floor(test_value), ",", np.ceil(test_value), "].")
            return None
        
    
    def PMF(self, klist):
        """
        Method implementing the Probability Mass Function (PMF) for the binomial distribution.
        """
        assert(len(klist)>0)
        for x in klist:
            assert(isinstance(x,int))
            assert(x>=0 and x<=self.n)
        pmf = [np.math.factorial(self.n) / (np.math.factorial(x) * np.math.factorial(self.n-x)) * np.power(self.p,x) * np.power(1-self.p,self.n-x) for x in klist]
        return pmf

    
    def CDF(self, klist):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the binomial distribution.
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
                tmp_sum += np.math.factorial(self.n) / (np.math.factorial(i) * np.math.factorial(self.n-i)) * np.power(self.p,i) * np.power(1-self.p,self.n-i)
                t.append(tmp_sum)
        else:
            for i in range(N):
                tmp_sum = [np.math.factorial(self.n) / (np.math.factorial(i) * np.math.factorial(self.n-i)) * np.power(self.p,i) * np.power(1-self.p,self.n-i) for i in range(np.floor(k_range[i]))].sum()
                t.append(tmp_sum)
        
        # Completing the calculation
        return np.array(t)
    
    
    
    
    
    
    