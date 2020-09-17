# __init__.py
__version__ = "0.1.0"
__author__ = "Fabien Nugier"


"""
The :mod:`scifin.classifier` module includes methods for classification and clustering.
"""

from .classifier import (euclidean_distance, dtw_distance, dtw_distance_matrix_from_ts,
                         kmeans_base_clustering, kmeans_advanced_clustering,
                         generate_random_classification, feature_importance_pvalues,
                         feature_importance_mdi, feature_importance_mda, feature_importance_mda)



