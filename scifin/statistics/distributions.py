# Created on 2020/7/24

# This module is for probability distributions.

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv, gamma, zeta



class Distribution:
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
        self.MAD = None # MAD = Mean Absolute Deviation
        
        # name (or nickname)
        self.name = name
    
    
    # Member functions
    
    def info(self):
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
        
class Normal(Distribution):
    """
    Class implementing the normal distribution of mean 'mu' and standard deviation 'sigma'.
    This class is inheriting from the class 'Distribution'.
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
        self.mean = mu
        self.variance = sigma*sigma
        self.std = np.sqrt(self.variance)
        self.skewness = 0.
        self.kurtosis = 3.
        
        # quantiles
        self.median = mu
        
        # others
        self.mode = mu
        self.MAD = sigma*np.sqrt(2/np.pi)
        self.entropy = (1/2) * np.log(2*np.pi*np.e*sigma*sigma)
        
        # name (or nickname)
        self.name = name
        
        
    def pdf(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Normal distribution.
        """
        pdf = np.exp(-(np.array(x)-self.mu) * (np.array(x)-self.mu) / (2 * self.sigma * self.sigma)) / (self.sigma * np.sqrt(2 * np.pi))
        return pdf

    def cdf(self, x):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the Normal distribution.
        """
        cdf = [(1/2) * (1 + erf((x_el - self.mu) / (self.sigma * np.sqrt(2)))) for x_el in x]
        return cdf
    
    def quantile(self, p):
        """
        Method returning the quantile associated to the Normal distribution.
        """
        assert(p>0 and p<1)
        return self.mu + self.sigma * np.sqrt(2) * erfinv(2*p-1)
    


class Uniform(Distribution):
    """
    Class implementing the uniform distribution taking a non-zero value between values 'a' and 'b>a'.
    This class is inheriting from the class 'Distribution'.
    """
    
    def __init__(self, a=0., b=1., name=""):
        """
        Initilialization function.
        """
        assert(a<b)
        
        # Type of distribution
        self.type = 'Uniform'
        self.support = '[a,b]'
        
        # parameters
        self.a = a
        self.b = b
                
        # moments
        self.mean = (a+b)/2
        self.variance = (b-a)**2 / 12
        self.std = np.sqrt(self.variance)
        self.skewness = 0
        self.kurtosis = 3. - 6./5
        
        # quantiles
        self.median = (a+b)/2
        
        # others
        self.mode = 'Any value between a and b.'
        self.entropy = np.log(b-a)
        
        # name (or nickname)
        self.name = name
        

    def pdf(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the uniform distribution.
        """
        pdf = []
        for i in x:
            if i>=self.a and i<=self.b:
                pdf.append(1/(self.b-self.a))
            else:
                pdf.append(0)
        return pdf
    
    
    def cdf(self, x):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the uniform distribution.
        """
        cdf = []
        for i in x:
            if i<self.a:
                cdf.append(0)
            elif i>=self.a and i<=self.b:
                cdf.append((i-self.a)/(self.b-self.a))
            elif i>self.b:
                cdf.append(1)
        return cdf
    

    
class Weibull(Distribution):
    """
    Class implementing the Weibull distribution with shape parameter 'k' (>0) and scale parameter 'lmbda' (>0). This class is inheriting from the class 'Distribution'.
    
    Note: default value for 'k' is 1, making it equivalent to an Exponential distribution.
    """
    
    def __init__(self, k=1, lmbda=1, name=""):
        """
        Initilialization function.
        """
        assert(k>0)
        assert(lmbda>0)
        
        # Type of distribution
        self.type = 'Weibull'
        self.support = 'R+'
        
        # parameters
        self.k = k
        self.lmbda = lmbda
                
        # moments
        self.mean = lmbda * gamma(1 + 1/k)
        self.variance = lmbda**2 * (gamma(1 + 2/k) - (gamma(1 + 1/k))**2)
        self.std = np.sqrt(self.variance)
        self.skewness = self.skewness_Weibull(k, lmbda)
        self.kurtosis = self.kurtosis_Weibull(k, lmbda)
        
        # quantiles
        self.median = lmbda * np.power(np.log(2), 1/k)
        
        # others
        self.mode = self.mode_Weibull(k, lmbda)
        self.entropy = np.euler_gamma * (1-1/k) + np.log(lmbda/k) + 1
        
        # name (or nickname)
        self.name = name
        
        
    def skewness_Weibull(self, k, lmbda):
        """
        Function computing the skewness of the Weibull distribution.
        """
        G1 = gamma(1 + 1/k)
        G2 = gamma(1 + 2/k)
        G3 = gamma(1 + 3/k)
        mu = lmbda * G1
        var = lmbda**2 * (G2 - G1**2)
        sig = np.sqrt(var)
        skew = (G3 * lmbda**3 - 3*mu*sig**2 - mu**3) / (sig**3)
        return skew
        
        
    def kurtosis_Weibull(self, k, lmbda):
        """
        Function computing the kurtosis of the Weibull distribution.
        """
        G1 = gamma(1 + 1/k)
        G2 = gamma(1 + 2/k)
        G3 = gamma(1 + 3/k)
        G4 = gamma(1 + 4/k)
        mu = lmbda * G1
        var = lmbda**2 * (G2 - G1**2)
        sig = np.sqrt(var)
        kurt = (G4 * lmbda**4 - 4 * self.skewness_Weibull(k,lmbda) * mu * sig**3 - 6 * mu**2 * sig**2 - mu**4) / (sig**4)
        return kurt
        
        
    def mode_Weibull(self, k, lmbda):
        """
        Function computing the mode of the Weibull distribution.
        """
        if k>1:
            return lmbda * np.power((k-1)/k, 1/k)
        else:
            return 0
        
        
    def pdf(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Weibull distribution.
        """
        pdf = []
        for i in x:
            if i>=0:
                pdf.append((self.k/self.lmbda) * np.power(i/self.lmbda,self.k-1) * np.exp(-np.power(i/self.lmbda,self.k)))
            else:
                pdf.append(0)
        return pdf

    
    def cdf(self, x):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the Weibull distribution.
        """
        cdf = []
        for i in x:
            if i>=0:
                cdf.append(1 - np.exp(-np.power(i/self.lmbda,self.k)))
            else:
                cdf.append(0)
        return cdf
    
    
    
    
    
class Rayleigh(Distribution):
    """
    Class implementing the Rayleigh distribution with scale parameter 'sigma'.
    This class is inheriting from the class 'Distribution'.
    """
    
    def __init__(self, sigma=1, name=""):
        """
        Initilialization function.
        """
        assert(sigma>0)
        
        # Type of distribution
        self.type = 'Rayleigh'
        self.support = 'R+'
        
        # parameters
        self.sigma = sigma
                
        # moments
        self.mean = sigma * np.sqrt(np.pi/2)
        self.variance = (4 - np.pi) / 2 * sigma**2
        self.std = np.sqrt(self.variance)
        self.skewness = 2 * np.sqrt(np.pi) * (np.pi - 3) / np.power(4 - np.pi, 3/2)
        self.kurtosis = 3 - (6*np.pi**2 - 24*np.pi + 16) / np.power(4 - np.pi, 2)
        
        # quantiles
        self.median = sigma * np.sqrt(2*np.log(2))
        
        # others
        self.mode = sigma
        self.entropy = 1 + np.log(sigma/np.sqrt(2)) + np.euler_gamma / 2
        
        # name (or nickname)
        self.name = name

        
    def pdf(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Rayleigh distribution.
        """
        pdf = []
        for i in x:
            if i>=0:
                pdf.append(i / (self.sigma**2) * np.exp(-i**2/(2*self.sigma**2)))
            else:
                pdf.append(0)
        return pdf

    
    def cdf(self, x):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the Rayleigh distribution.
        """
        cdf = []
        for i in x:
            if i>=0:
                cdf.append(1 - np.exp(-i**2/(2*self.sigma**2)))
            else:
                cdf.append(0)
        return cdf


    
class Exponential(Distribution):
    """
    Class implementing the Exponential distribution with rate parameter 'lmbda' (>0).
    This class is inheriting from the class 'Distribution'.
    """
    
    def __init__(self, lmbda=1 , name=""):
        """
        Initilialization function.
        """
        assert(lmbda>0)
        
        # Type of distribution
        self.type = 'Exponential'
        self.support = 'R+'
        
        # parameters
        self.lmbda = lmbda
                
        # moments
        self.mean = 1/lmbda
        self.variance = 1 / lmbda**2
        self.std = np.sqrt(self.variance)
        self.skewness = 2
        self.kurtosis = 9
        
        # quantiles
        self.median = np.log(2) / lmbda
        
        # others
        self.mode = 0
        self.entropy = 1 - np.log(lmbda)
        
        # name (or nickname)
        self.name = name


    def pdf(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Exponential distribution.
        """
        pdf = []
        for i in x:
            if i>=0:
                pdf.append(self.lmbda * np.exp(- self.lmbda * i))
            else:
                pdf.append(0)
        return pdf

    
    def cdf(self, x):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the Exponential distribution.
        """
        cdf = []
        for i in x:
            if i>=0:
                cdf.append(1 - np.exp(- self.lmbda * i))
            else:
                cdf.append(0)
        return cdf
    
    
    def quantile(self, p):
        """
        Method returning the quantile associated to the Exponential distribution.
        """
        assert(p>0 and p<1)
        return - np.log(1 - p) / self.lmbda
    
    
    
class Gumbel(Distribution):
    """
    Class implementing the Gumbel distribution with mode 'mu' and parameter 'beta' (>0).
    This class is inheriting from the class 'Distribution'.
    """
    
    def __init__(self, mu=0, beta=1 , name=""):
        """
        Initilialization function.
        """
        assert(beta>0)
        
        # Type of distribution
        self.type = 'Gumbel'
        self.support = 'R'
        
        # parameters
        self.mu = mu
        self.beta = beta
                
        # moments
        self.mean = mu + beta * np.euler_gamma
        self.variance = np.pi**2 * beta**2 / 6
        self.std = np.sqrt(self.variance)
        self.skewness = 12 * np.sqrt(6) * zeta(3) / np.power(np.pi, 3)
        self.kurtosis = 3 + 12/5
        
        # quantiles
        self.median = mu - beta * np.log(np.log(2))
        
        # others
        self.mode = mu
        self.entropy = np.log(beta) + np.euler_gamma + 1
        
        # name (or nickname)
        self.name = name

        
    def pdf(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Gumbel distribution.
        """
        z = (np.array(x) - self.mu) / self.beta
        pdf = np.exp(-z-np.exp(-z)) / self.beta
        return pdf

    
    def cdf(self, x):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the Gumbel distribution.
        """
        z = (np.array(x) - self.mu) / self.beta
        cdf = np.exp(-np.exp(-z))
        return cdf
    
    
    def quantile(self, p):
        """
        Method returning the quantile associated to the Gumbel distribution.
        """
        assert(p>0 and p<1)
        return self.mu - self.beta * np.log(-np.log(p))


    
    
class Laplace(Distribution):
    """
    Class implementing the Laplace distribution with mean 'mu' and scale parameter 'b' (>0).
    This class is inheriting from the class 'Distribution'.
    """
    
    def __init__(self, mu=0, b=1 , name=""):
        """
        Initilialization function.
        """
        assert(b>0)
        
        # Type of distribution
        self.type = 'Laplace'
        self.support = 'R'
        
        # parameters
        self.mu = mu
        self.b = b
                
        # moments
        self.mean = mu
        self.variance = 2 * b**2
        self.std = np.sqrt(self.variance)
        self.skewness = 0
        self.kurtosis = 6
        
        # quantiles
        self.median = mu
        
        # others
        self.mode = mu
        self.entropy = np.log(2 * b * np.e)
        
        # name (or nickname)
        self.name = name


    def pdf(self, x):
        """
        Method implementing the Probability Density Function (PDF) for the Laplace distribution.
        """
        pdf = np.exp(-np.abs(np.array(x)-self.mu)/self.b) / (2*self.b)
        return pdf

    
    def cdf(self, x):
        """
        Method implementing the Cumulative Distribution Function (CDF) for the Laplace distribution.
        """
        cdf = []
        for i in x:
            if i <= self.mu:
                cdf.append(0.5 * np.exp((i-self.mu)/self.b))
            else:
                cdf.append(1 - 0.5 * np.exp(-(i-self.mu)/self.b))
        return cdf
    
    
    def quantile(self, p):
        """
        Method returning the quantile associated to the Laplace distribution.
        """
        assert(p>0 and p<1)
        if p <= 1/2:
            return self.mu + self.b * np.log(2*p)
        else:
            return self.mu - self.b * np.log(2-2*p)
    
    
    
    
    
# DISCRETE DISTRIBUTIONS
    
class Poisson(Distribution):
    """
    Class implementing the Poisson distribution with expected rate of event occurence 'lambda'.
    This class is inheriting from the class 'Distribution'.
    
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
        self.mean = lmbda
        self.variance = lmbda
        self.std = np.sqrt(self.variance)
        self.skewness = 1./np.sqrt(lmbda)
        self.kurtosis = 3. + 1./np.sqrt(lmbda)
        
        # quantiles
        self.median = np.floor(lmbda + 1/3 - 0.02/lmbda)
        
        # others
        self.k_max = k_max
        self.mode = np.floor(lmbda)
        self.entropy = self.entropy_Poisson(lmbda)
        
        # name (or nickname)
        self.name = name
    
    
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
        
        
    def pmf(self, klist):
        """
        Method implementing the Probability Mass Function (PMF) for the Poisson distribution.
        """
        assert(len(klist)>0)
        
        for x in klist:
            assert(isinstance(x,int))
        pmf = [np.power(self.lmbda,x) * np.exp(-self.lmbda) / np.math.factorial(x) for x in klist]
        return pmf

    
    def cdf(self, klist):
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
    
    
    
    
    
    
    
class Binomial(Distribution):
    """
    Class implementing the binomial distribution of "successes" having probability of success 'p' in a sequence of 'n' independent experiments (number of trials). I also applies to drawing an element with probability p in a population of n elements, with replacement after draw. This class is inheriting from the class 'Distribution'.
    
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
        self.mean = n*p
        self.variance = n*p*(1-p)
        self.std = np.sqrt(self.variance)
        self.skewness = ((1-p)-p)/np.sqrt(n*p*(1-p))
        self.kurtosis = 3. + (1-6*p*(1-p))/(n*p*(1-p))
        
        # quantiles
        self.median = self.median_Binomial(n, p)
        
        # others
        self.mode = self.mode_Binomial(n, p)
        self.entropy = (1/2) * np.log2(2 * np.pi * np.e * n*p*(1-p))
        
        # name (or nickname)
        self.name = name
        

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
        
    
    def pmf(self, klist):
        """
        Method implementing the Probability Mass Function (PMF) for the binomial distribution.
        """
        assert(len(klist)>0)
        for x in klist:
            assert(isinstance(x,int))
            assert(x>=0 and x<=self.n)
        pmf = [np.math.factorial(self.n) / (np.math.factorial(x) * np.math.factorial(self.n-x)) * np.power(self.p,x) * np.power(1-self.p,self.n-x) for x in klist]
        return pmf

    
    def cdf(self, klist):
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
    
    
    
    
    
    
    