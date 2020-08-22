# __init__.py
__version__ = "0.1.0"
__author__ = "Fabien Nugier"

"""
The :mod:`scifin.statistics` module includes methods for statistics.
"""

from .distributions  import standard_normal_pdf, standard_normal_cdf, standard_normal_quantile, \
                            Distribution, Normal, Uniform, Weibull, Rayleigh, \
                            Exponential, Gumbel, Laplace, Levy, Cauchy, \
                            Poisson, Binomial



