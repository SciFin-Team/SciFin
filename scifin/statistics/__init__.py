# __init__.py
__version__ = "0.1.0"
__author__ = "Fabien Nugier"

"""
The :mod:`scifin.statistics` module includes methods for statistics.
"""

from .distributions  import (upper_incomplete_gamma,
                             standard_normal_pdf, standard_normal_cdf, standard_normal_quantile,
                             Distribution, Normal, Uniform, Weibull, Rayleigh,
                             Exponential, Gumbel, Laplace, Levy, Cauchy,
                             Poisson, Binomial)

from .statistics import (random_covariance_matrix, covariance_to_correlation, correlation_to_covariance,
                         get_subcovariance, random_block_covariance, random_block_correlation,
                         covariance_from_ts, denoise_covariance, eigen_value_vector,
                         marcenko_pastur_pdf, marcenko_pastur_loss, marcenko_pastur_fit_params,
                         distance_from_vectors, pearson_correlation, distance_from_pearson,
                         distance_from_abs_pearson, entropy_info)

