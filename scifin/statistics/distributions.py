# Created on 2020/7/24

# This module is for probability distributions.

# Standard library imports
from typing import TypeVar, Generic, Union

# Third party imports
import numpy as np
from scipy.special import erf, erfinv, gamma, zeta, gammaincc
from typeguard import typechecked

# Local application imports
# /

# New Variables Types
Distribution = TypeVar('Distribution')

#---------#---------#---------#---------#---------#---------#---------#---------#---------#


@typechecked
def upper_incomplete_gamma(a: Union[int, float],
                           z: Union[int, float, list, np.ndarray]
                           ) -> Union[float, list, np.ndarray]:
    """
    Implements the Upper Incomplete Gamma function with parameter a
    at the value z.

    Parameters
    ----------
    a: int, float
      Evaluation parameter.
    z: int, float, list, np.ndarray
      Evaluation variable.

    Returns
    -------
    float
      Evaluation of the upper incomplete gamma function.

    Notes
    -----
      For more information, consult the following:
      https://en.wikipedia.org/wiki/Incomplete_gamma_function
    """

    # Checks
    if a < 0:
        raise AssertionError("Argument a must be positive.")

    return gamma(a) * gammaincc(a, z)


@typechecked
class Distribution(Generic[Distribution]):
    """
    Abtract class that reates a statistical distribution.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    MAD : float
      Mean Absolute Deviation of the distribution.
    name : str
      Name of nickname given to the distribution.
    """

    @typechecked
    def __init__(self, name: str="") -> None:
        """
        Initializes the distribution.
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
    
    def info(self) -> None:
        """
        Prints the most relevant information about the distribution.
        """
        print("Name: \t\t", self.name)
        print("Type: \t\t", self.type)
        print("Mean: \t\t", self.mean)
        print("Variance: \t", self.variance)
        print("Std. Dev.: \t", self.std)
        print("Skewness: \t", self.skewness)
        print("Kurtosis: \t", self.kurtosis)
        print("Median: \t", self.median)
        print("Mode: \t\t", self.mode)
        print("Entropy: \t", self.entropy)
        
        return None



# UTILITY FUNCTIONS

@typechecked
def check_type_x(x: Union[int, float, list, np.ndarray]) -> None:
    """
    Checks if x has the right type.

    Parameters
    ----------
    x : int, long, float, list, np.ndarray
      Argument to check.

    Returns
    -------
    None
      None
    """

    if isinstance(x, list):
        if any(not isinstance(xi, (int, float)) for xi in x):
            raise AssertionError("Some element of x is neither int nor float.")

    return None


@typechecked
def check_type_p(p: Union[int, float, list, np.ndarray]) -> None:
    """
    Checks if p has the right type.

    Parameters
    ----------
    p : int, float, list, np.ndarray
    Argument to check.

    Returns
    -------
    None
      None
    """

    if isinstance(p, list):
        if any(not (isinstance(pi, (int, float)) and 0 <= pi <= 1) for pi in p):
            raise AssertionError("Some element in p is inappropriate (either not int nor float, or not in [0,1].")

    return None


@typechecked
def check_type_k(k: Union[int, list, range, np.ndarray]) -> None:
    """
    Checks if k has the right type.

    Parameters
    ----------
    k : int, list, np.ndarray
    Argument to check.

    Returns
    -------
    None
      None
    """

    if isinstance(k, list):
        if any(not (isinstance(ki,int)) for ki in k):
            raise AssertionError("Some element in k is not int.")

    return None


@typechecked
def initialize_input(xorp: Union[int, float, list, np.ndarray]) -> np.ndarray:
    """
    Returns an np.ndarray from x or p.

    Parameters
    ----------
    xorp : int, float, list, np.ndarray
      Argument to transform.

    Returns
    -------
    np.ndarray
      The transformed input.
    """

    if isinstance(xorp, np.ndarray):
        return xorp
    elif isinstance(xorp, list):
        return np.array(xorp)
    elif isinstance(xorp, (int, float)):
        return np.array([xorp])
    elif isinstance(xorp, range):
        return np.array(xorp)




# CONTINUOUS DISTRIBUTIONS

@typechecked
def standard_normal_pdf(x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
    """
    Implements the Probability Density Function (PDF) for the Standard Normal distribution.

    Arguments
    ---------
    x : int, float, list, np.ndarray
      Evaluation value(s).

    Returns
    -------
    float, np.ndarray
      Standard normal PDF for evaluation value(s).
    """

    # Check
    check_type_x(x)
    x = initialize_input(x)

    # Compute
    pdf = np.exp(- x**2 / 2) / np.sqrt(2*np.pi)

    # Return
    if len(pdf) == 1:
        return pdf[0]
    else:
        return pdf


@typechecked
def standard_normal_cdf(x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
    """
    Implements the Cumulative Distribution Function (CDF) for the Standard Normal distribution.

    Arguments
    ---------
    x : int, float, list, np.ndarray
      Evaluation value(s).

    Returns
    -------
    float, np.ndarray
      Standard normal CDF for evaluation value(s).
    """

    # Check
    check_type_x(x)
    x = initialize_input(x)

    # Compute
    cdf = (1/2) * ( 1 + erf(x / np.sqrt(2)) )

    # Return
    if len(cdf)==1:
        return cdf[0]
    else:
        return cdf


@typechecked
def standard_normal_quantile(p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
    """
    Returns the quantile associated to the Standard Normal distribution.

    Arguments
    ---------
    x : int, float, list, np.ndarray
      Evaluation value(s).

    Returns
    -------
    float, np.ndarray
      Standard normal PDF for evaluation value(s).
    """

    # Check
    check_type_p(p)
    p = initialize_input(p)

    # Compute
    quantile = np.sqrt(2) * erfinv(2*p-1)

    # Return
    if len(quantile)==1:
        return quantile[0]
    else:
        return quantile


@typechecked
class Normal(Distribution):
    """
    Implements the normal distribution of mean 'mu' and standard deviation 'sigma'.

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    mu : float
      Mean parameter of the distribution.
    sigma : float
      Standard deviation parameter (>0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    MAD : float
      Mean Absolute Deviation of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    """

    def __init__(self, mu: float=0., sigma: float=1., name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not sigma>0:
            raise AssertionError('Value of sigma must be non-zero and positive.')

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

    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the Normal distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        pdf = np.exp(-(x-self.mu)**2 / (2 * self.sigma**2)) / (self.sigma * np.sqrt(2 * np.pi))

        # Return
        if len(pdf) == 1:
            return pdf[0]
        else:
            return pdf

    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the Normal distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        cdf = (1/2) * (1 + erf((x - self.mu) / (self.sigma * np.sqrt(2))))

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf
    
    def quantile(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the quantile associated to the Normal distribution.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        quantile = self.mu + self.sigma * np.sqrt(2) * erfinv(2*p-1)

        # Return
        if len(quantile)==1:
            return quantile[0]
        else:
            return quantile
    
    # Alias method
    var = quantile
        
    def cvar(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the Conditional Value At Risk (CVaR) of the Normal distribution
        for a certain probability p.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        cvar = self.mu + self.sigma * standard_normal_pdf(standard_normal_quantile(p)) / p

        # Return
        if len(p)==1:
            return cvar[0]
        else:
            return cvar


@typechecked
class Uniform(Distribution):
    """
    Implements the uniform distribution taking a non-zero value
    between values 'a' and 'b>a'.
    
    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    a : float
      Left parameter of the distribution.
    b : float
      Right parameter of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
      
    Notes
    -----
      Requires b>a.
    """
    
    def __init__(self, a: float=0., b: float=1., name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not (a<b):
            raise AssertionError("a < b is required.")
        
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
        
    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the uniform distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        pdf = []
        for i in x:
            if i >= self.a and i <= self.b:
                pdf.append(1/(self.b-self.a))
            else:
                pdf.append(0)

        # Return
        if len(pdf) == 1:
            return pdf[0]
        else:
            return np.array(pdf)

    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the uniform distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        cdf = []
        for i in x:
            if i < self.a:
                cdf.append(0)
            elif i >= self.a and i <= self.b:
                cdf.append((i-self.a)/(self.b-self.a))
            elif i > self.b:
                cdf.append(1)

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return np.array(cdf)

    
@typechecked
class Weibull(Distribution):
    """
    Implements the Weibull distribution with shape parameter 'k' (>0)
    and scale parameter 'lmbda' (>0).
    
    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    k : float
      Shape parameter (>0) of the distribution.
    lmbda : float
      Scale parameter (>0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    
    Notes
    -----
      Default value for 'k' is 1, making it equal to an Exponential distribution.
    """
    
    def __init__(self, k: float=1, lmbda: float=1, name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not (k>0):
            raise AssertionError("k>0 is required.")
        if not (lmbda>0):
            raise AssertionError("lmbda>0 is required.")
        
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
        self.skewness = self.get_skewness(k, lmbda)
        self.kurtosis = self.get_kurtosis(k, lmbda)
        
        # quantiles
        self.median = lmbda * np.power(np.log(2), 1/k)
        
        # others
        self.mode = self.get_mode(k, lmbda)
        self.entropy = np.euler_gamma * (1-1/k) + np.log(lmbda/k) + 1
        
        # name (or nickname)
        self.name = name
        
    def get_skewness(self, k: float, lmbda: float) -> float:
        """
        Computes the skewness of the Weibull distribution.
        """
        g1 = gamma(1 + 1/k)
        g2 = gamma(1 + 2/k)
        g3 = gamma(1 + 3/k)
        mu = lmbda * g1
        var = lmbda**2 * (g2 - g1**2)
        sig = np.sqrt(var)
        skew = (g3 * lmbda**3 - 3*mu*sig**2 - mu**3) / (sig**3)
        return skew
        
    def get_kurtosis(self, k: float, lmbda: float) -> float:
        """
        Computes the kurtosis of the Weibull distribution.
        """
        g1 = gamma(1 + 1/k)
        g2 = gamma(1 + 2/k)
        g3 = gamma(1 + 3/k)
        g4 = gamma(1 + 4/k)
        mu = lmbda * g1
        var = lmbda**2 * (g2 - g1**2)
        sig = np.sqrt(var)
        kurt = (g4 * lmbda**4 - 4 * self.get_skewness(k,lmbda) 
                   * mu * sig**3 - 6 * mu**2 * sig**2 - mu**4) / (sig**4)
        return kurt
        
    def get_mode(self, k: float, lmbda: float) -> float:
        """
        Computes the mode of the Weibull distribution.
        """
        if k>1:
            return lmbda * np.power((k-1)/k, 1/k)
        else:
            return 0
        
    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the Weibull distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        pdf = []
        for i in x:
            if i>=0:
                pdf.append((self.k/self.lmbda) * np.power(i/self.lmbda,self.k-1) 
                                               * np.exp(-np.power(i/self.lmbda,self.k)))
            else:
                pdf.append(0)

        # Return
        if len(pdf)==1:
            return pdf[0]
        else:
            return np.array(pdf)

    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the Weibull distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        cdf = []
        for i in x:
            if i>=0:
                cdf.append(1 - np.exp(-np.power(i/self.lmbda,self.k)))
            else:
                cdf.append(0)

        # Return
        if len(cdf) == 1:
            return cdf[0]
        else:
            return np.array(cdf)
    
    def quantile(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the quantile associated to the Weibull distribution.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        quantile = self.lmbda * np.power(-np.log(1-p), 1/self.k)

        # Return
        if len(quantile)==1:
            return quantile[0]
        else:
            return quantile
    
    # Alias method
    var = quantile
        
    def cvar(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the Conditional Value At Risk (CVaR) of the Weibull distribution
        for a certain probability p.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        cvar = (self.lmbda/(1-p)) * upper_incomplete_gamma(1 + 1/self.k, -np.log(1-p))

        # Return
        if len(cvar)==1:
            return cvar[0]
        else:
            return cvar


@typechecked
class Rayleigh(Distribution):
    """
    Implements the Rayleigh distribution with scale parameter 'sigma'.

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    sigma : float
      Scale parameter (>0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    """
    
    def __init__(self, sigma: float=1, name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not (sigma>0):
            raise AssertionError("sigma>0 is required.")
        
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

    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the Rayleigh distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        pdf = []
        for i in x:
            if i>=0:
                pdf.append(i / (self.sigma**2) * np.exp(-i**2/(2*self.sigma**2)))
            else:
                pdf.append(0)

        # Return
        if len(pdf)==1:
            return pdf[0]
        else:
            return pdf

    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the Rayleigh distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        cdf = []
        for i in x:
            if i>=0:
                cdf.append(1 - np.exp(-i**2/(2*self.sigma**2)))
            else:
                cdf.append(0)

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf


@typechecked
class Exponential(Distribution):
    """
    Implements the Exponential distribution with rate parameter 'lmbda' (>0).

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    lmbda : float
      Rate parameter (>0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    """
    
    def __init__(self, lmbda: float=1, name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not (lmbda>0):
            raise AssertionError("lmbda>0 is required.")
        
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

    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the Exponential distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        pdf = []
        for i in x:
            if i>=0:
                pdf.append(self.lmbda * np.exp(- self.lmbda * i))
            else:
                pdf.append(0)

        # Return
        if len(pdf)==1:
            return pdf[0]
        else:
            return pdf

    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the Exponential distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        cdf = []
        for i in x:
            if i>=0:
                cdf.append(1 - np.exp(- self.lmbda * i))
            else:
                cdf.append(0)

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf
    
    def quantile(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the quantile associated to the Exponential distribution.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        quantile = - np.log(1 - p) / self.lmbda

        # Return
        if len(quantile)==1:
            return quantile[0]
        else:
            return quantile
    
    # Alias method
    var = quantile
        
    def cvar(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the Conditional Value At Risk (CVaR) of the Exponential distribution
        for a certain probability p.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        cvar = (-np.log(1-p)+1)/self.lmbda

        # Return
        if len(cvar)==1:
            return cvar[0]
        else:
            return cvar
    

@typechecked
class Gumbel(Distribution):
    """
    Implements the Gumbel distribution with mode 'mu' and parameter 'beta' (>0).

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    mu : float
      Mode parameter of the distribution.
    beta : float
      Initialization parameter (>0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    MAD : float
      Mean Absolute Deviation of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    """
    
    def __init__(self, mu: float=0, beta: float=1, name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not (beta>0):
            raise AssertionError("beta>0 is required.")
        
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

    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the Gumbel distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        z = (np.array(x) - self.mu) / self.beta
        pdf = np.exp(-z-np.exp(-z)) / self.beta

        # Return
        if len(pdf)==1:
            return pdf[0]
        else:
            return pdf

    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the Gumbel distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        z = (np.array(x) - self.mu) / self.beta
        cdf = np.exp(-np.exp(-z))

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf
    
    def quantile(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the quantile associated to the Gumbel distribution.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        quantile = self.mu - self.beta * np.log(-np.log(p))

        # Return
        if len(quantile)==1:
            return quantile[0]
        else:
            return quantile

    # Alias method
    var = quantile
    

@typechecked
class Laplace(Distribution):
    """
    Implements the Laplace distribution with mean 'mu' and scale parameter 'b' (>0).

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    mu : float
      Mean parameter of the distribution.
    b : float
      Scale parameter (>0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    """
    
    def __init__(self, mu=0, b=1 , name=""):
        """
        Initializes the distribution.
        """
        if not (b>0):
            raise AssertionError("b>0 is required.")
        
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

    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the Laplace distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        pdf = np.exp(-np.abs(np.array(x)-self.mu)/self.b) / (2*self.b)

        # Return
        if len(pdf)==1:
            return pdf[0]
        else:
            return pdf

    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the Laplace distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        cdf = []
        for i in x:
            if i <= self.mu:
                cdf.append(0.5 * np.exp((i-self.mu)/self.b))
            else:
                cdf.append(1 - 0.5 * np.exp(-(i-self.mu)/self.b))

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf
    
    def quantile(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the quantile associated to the Laplace distribution.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        quantile = []
        for pi in p:
            if pi <= 1/2:
                quantile.append(self.mu + self.b * np.log(2*p))
            else:
                quantile.append(self.mu - self.b * np.log(2-2*p))

        # Return
        if len(quantile)==1:
            return quantile[0]
        else:
            return quantile

    # Alias method
    var = quantile
        
    def cvar(self, p: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Returns the Conditional Value At Risk (CVaR) of the Laplace distribution
        for a certain probability p.
        """

        # Check
        check_type_p(p)
        p = initialize_input(p)

        # Compute
        cvar = []
        for pi in p:
            if pi <= 1/2:
                cvar.append(self.mu + self.b * p / (1-p) * (1 - np.log(2*p)))
            else:
                cvar.append(self.mu + self.b * (1 - np.log(2-2*p)))

        # Return
        if len(cvar)==1:
            return cvar[0]
        else:
            return cvar


@typechecked
class Levy(Distribution):
    """
    Implements the Lévy distribution with location parameter 'mu'
    and scale parameter 'c' (>0).

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    mu : float
      Location parameter of the distribution.
    c : float
      Scale parameter (>0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    """
    
    def __init__(self, mu: float=0, c: float=1, name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not (c>0):
            raise AssertionError("c>0 is required.")
        
        # Type of distribution
        self.type = 'Levy'
        self.support = '[mu, Infinity)'
        
        # parameters
        self.mu = mu
        self.c = c
                
        # moments
        self.mean = 'Infinity'
        self.variance = 'Infinity'
        self.std = 'Infinity'
        self.skewness = None
        self.kurtosis = None
        
        # quantiles
        self.median = mu + (c/2) * (erfinv(1/2))**2
        
        # others
        self.mode = mu + c/3
        self.entropy = (1 + 3 * np.euler_gamma + np.log(16*np.pi*c**2))
        
        # name (or nickname)
        self.name = name

    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the Lévy distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        pdf = []
        for i in x:
            if i > self.mu:
                pdf.append( np.sqrt(self.c/(2*np.pi)) 
                               * np.exp( -self.c / (2*(i-self.mu)) )
                               / np.power(i-self.mu,3/2)
                          )
            else:
                pdf.append(0)

        # Return
        if len(pdf)==1:
            return pdf[0]
        else:
            return pdf

    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the Lévy distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        cdf = []
        for i in x:
            if i > self.mu:
                cdf.append( 1 - erf(np.sqrt( self.c/(2*(i-self.mu)) )) )
            else:
                cdf.append(0)

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf


@typechecked
class Cauchy(Distribution):
    """
    Implements the Cauchy distribution with mode/median 'a'
    and scale parameter 'b' (>0).

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    a : float
      Mean/mode parameter of the distribution.
    b : float
      Scale parameter (>0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    """
    
    def __init__(self, a: float=0, b: float=1 , name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not (b>0):
            raise AssertionError("b>0 is required.")
        
        # Type of distribution
        self.type = 'Cauchy'
        self.support = 'R'
        
        # parameters
        self.a = a
        self.b = b
                
        # moments
        self.mean = None
        self.variance = None
        self.std = None
        self.skewness = None
        self.kurtosis = None
        
        # quantiles
        self.median = a
        
        # others
        self.mode = a
        self.entropy = np.log(4*np.pi*b)
        
        # name (or nickname)
        self.name = name

    def pdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Density Function (PDF)
        for the Cauchy distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        z = (np.array(x) - self.a) / self.b
        pdf = 1 / (np.pi * self.b) / (1 + z**2)

        # Return
        if len(pdf)==1:
            return pdf[0]
        else:
            return pdf
    
    def cdf(self, x: Union[int, float, list, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF)
        for the Cauchy distribution.
        """

        # Check
        check_type_x(x)
        x = initialize_input(x)

        # Compute
        z = (np.array(x) - self.a) / self.b
        cdf = 0.5 + np.arctan(z) / np.pi

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf
        

    
# DISCRETE DISTRIBUTIONS

@typechecked
class Poisson(Distribution):
    """
    Implements the Poisson distribution
    with expected rate of event occurence 'lambda' (>=0).

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    lmbda : float
      Rate of occurence parameter (>=0) of the distribution.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    k_max : int, >0
      Limit of summation in entropy calculation.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    
    Notes
    -----
      We use a value k_max to set the limit of summation
      for the entropy calculation.
    """
    
    def __init__(self, lmbda: float=0., k_max: int=1000, name: str="") -> None:
        """
        Initializes the distribution.
        """
        if not (lmbda>0):
            raise AssertionError("lmbda>0 is required.")
        if not isinstance(k_max,int) and not (k_max>0):
            raise AssertionError("k_max must be integer and k_max>0.")
        
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
        self.entropy = self.get_entropy(lmbda)
        
        # name (or nickname)
        self.name = name
    
    def get_entropy(self, lmbda: float) -> float:
        """
        Computes the entropy for the Poisson distribution.
        """
        tmp_sum = 0.
        for k in range(self.k_max):
            contrib = np.power(lmbda,k) \
                      * np.log(np.math.factorial(k)) / np.math.factorial(k)
            if contrib < 1.e-15:
                tmp_sum += contrib
                break
        if k==self.k_max:
            print("Careful. Sum probably did not converge.")

        return lmbda * (1-np.log(lmbda)) + np.exp(-lmbda) * tmp_sum
        
    def pmf(self, k: Union[int, list, range, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Mass Function (PMF) for the Poisson distribution.
        """

        # Check
        check_type_k(k)
        k = initialize_input(k)

        # Compute
        factorials = np.array([np.math.factorial(ki) for ki in k])
        pmf = np.power(self.lmbda, k) * np.exp(-self.lmbda) / factorials

        # Return
        if len(pmf)==1:
            return pmf[0]
        else:
            return pmf

    def cdf(self, k: Union[int, list, range, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF) for the Poisson distribution.
        """

        # Check
        check_type_k(k)
        k = initialize_input(k)
        N = len(k)
            
        # Check if k's are consecutive
        ks_consecutive = True
        for i in range(N-1):
            if (k[i+1] != k[i]+1):
                ks_consecutive = False
                break
        
        # But if there is only one element
        # Consider k's not-consecutive case
        if N==1:
            ks_consecutive = False
        
        # Compute the list of elements to sum
        t = []
        k_range = np.floor(k)
        if ks_consecutive==True:
            tmp_sum = sum([ np.power(self.lmbda,i) / np.math.factorial(i) 
                            for i in range(int(np.floor(k_range[0]+1))) ])
            t.append(tmp_sum)
            for i in range(int(k_range[0]+1),int(k_range[-1]+1),1):
                tmp_sum += np.power(self.lmbda,i) / np.math.factorial(i)
                t.append(tmp_sum)
        else:
            for i in range(N):
                tmp_sum = sum([ np.power(self.lmbda,i) / np.math.factorial(i) 
                            for i in range(int(np.floor(k_range[i]+1))) ])
                t.append(tmp_sum)
        
        # Completing the calculation
        cdf = np.exp(-self.lmbda) * np.array(t)

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf
    

@typechecked
class Binomial(Distribution):
    """
    Implements the binomial distribution of "successes" having
    probability of success 'p' in a sequence of 'n' independent
    experiments (number of trials).
    
    I also applies to drawing an element with probability p in
    a population of n elements, with replacement after draw.

    This class inherits from the parent class 'Distribution'.
    
    Attributes
    ----------
    type : str
      Represents the type of distribution.
    support : str
      Represents the support of the distribution.
    p : float in [0,1]
      Probability of success for a trial.
    n : int
      Number of independent experiments.
    q : float in [0,1]
      Probability of failure, i.e. q = 1 - p.
    mean : float
      Mean of the distribution.
    variance : float
      Variance of the distribution.
    std : float
      Standard deviation of the distribution.
    skewness : float
      Skewness of the distribution.
    kurtosis : float
      Kurtosis of the distribution.
    median : float
      Median of the distribution.
    mode : float
      Mode of the distribution.
    entropy : float
      Entropy of the distribution.
    name : str
      Name of nickname given to the distribution.
    
    Notes
    -----
      Default value for n is taken to be 1,
      hence corresponding to the Bernoulli distribution.
      
      Computation of entropy is only an approximation
      valid at order O(1/n).
    """
    
    def __init__(self, n: int=1, p: float=0.5, name: str="") -> None:
        """
        Initializes the distribution.
        """
        # Checks
        if not isinstance(n,int) or not (n>=0):
            raise AssertionError("n must be integer and n>=0.")
        if not (0. <= p <= 1.):
            raise AssertionError("p must be in [0,1].")
        
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
        self.median = self.get_median(n, p)
        
        # others
        self.mode = self.get_mode(n, p)
        self.entropy = (1/2) * np.log2(2 * np.pi * np.e * n*p*(1-p))
        
        # name (or nickname)
        self.name = name
        
    def get_mode(self, n: int, p: float) -> int:
        """
        Computes the mode of the Binomial distribution.
        """
        test_value = (n+1)*p
        if test_value == 0 or isinstance(test_value,int) == False:
            return int(np.floor(test_value))
        elif test_value in range(1,n,1):
            print("Binomial distribution for these values has two modes.")
            return (test_value, test_value-1)
        elif test_value == n+1:
            return n

    def get_median(self, n: int, p: float) -> Union[float, None]:
        """
        Partially computes the median of the Binomial distribution.
        """
        test_value = n*p
        if test_value==int(test_value):
            return test_value
        else:
            print("Median has a value in interval [", np.floor(test_value), ",",
                  np.ceil(test_value), "].")
            return None
    
    def pmf(self, k: Union[int, list, range, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Probability Mass Function (PMF) for the binomial distribution.
        """

        # Check
        check_type_k(k)
        k = initialize_input(k)
        for ki in k:
            assert(ki>=0 and ki<=self.n)

        # Compute
        pmf = [ np.math.factorial(self.n)
                / (np.math.factorial(x) * np.math.factorial(self.n-x)) 
                * np.power(self.p,x) * np.power(1-self.p,self.n-x) 
                for x in k ]

        # Return
        if len(pmf)==1:
            return pmf[0]
        else:
            return pmf
    
    def cdf(self, k: Union[int, list, range, np.ndarray]) -> Union[float, list, np.ndarray]:
        """
        Implements the Cumulative Distribution Function (CDF) for the binomial distribution.
        """

        # Check
        check_type_k(k)
        k = initialize_input(k)
        N = len(k)
            
        # Check if k's are consecutive
        ks_consecutive = True
        for i in range(N-1):
            if (k[i+1] != k[i]+1):
                ks_consecutive = False
                break
        
        # But if there is only one element
        # Consider k's not-consecutive case
        if N==1:
            ks_consecutive = False
        
        # Compute the list of elements to sum
        t = []
        k_range = np.floor(k)
        if ks_consecutive==True:
            tmp_sum = sum([ np.math.factorial(self.n) \
                           / (np.math.factorial(i) * np.math.factorial(self.n-i)) \
                           * np.power(self.p,i) * np.power(1-self.p,self.n-i) 
                            for i in range(int(np.floor(k_range[0]+1))) ])
            t.append(tmp_sum)
            for i in range(int(k_range[0]+1),int(k_range[-1]+1),1):
                tmp_sum += np.math.factorial(self.n) \
                           / (np.math.factorial(i) * np.math.factorial(self.n-i)) \
                           * np.power(self.p,i) * np.power(1-self.p,self.n-i)
                t.append(tmp_sum)
        else:
            for i in range(N):
                tmp_sum = sum([ np.math.factorial(self.n) 
                            / (np.math.factorial(i) * np.math.factorial(self.n-i)) 
                            * np.power(self.p,i) * np.power(1-self.p,self.n-i) 
                            for i in range(int(np.floor(k_range[i]+1))) ])
                t.append(tmp_sum)
        
        # Completing the calculation
        cdf = np.array(t)

        # Return
        if len(cdf)==1:
            return cdf[0]
        else:
            return cdf
    
#---------#---------#---------#---------#---------#---------#---------#---------#---------#