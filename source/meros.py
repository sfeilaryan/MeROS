"""
A compact wrapper for meta-recognition and more specifically open-set classification functionality.
The main implementations are summarized below along with their respective references if applicable.

- Model Meta-Recognition Calibration Based on EVT - Centroid(s) and Weibull Distance Models - Tools

Reference:
Meta-Recognition: The Theory and Practice of Recognition Score Analysis
Authors: Walter J. Scheirer, Anderson Rocha, Ross Michaels, and Terrance E. Boult
Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)
Volume: 33, Issue: 8, Pages: 1689-1695
Year: 2011

- Computing at Run Time Probability of Unseen Unknown - Reject Novelty - Open-Space Risk Control - OpenMax

Reference:
Towards Open Set Deep Networks
Authors: Abhijit Bendale and Terrance Boult
Presented at: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
Published by: IEEE

- Determining at Run Time the Centroid(s) of Each Class Using K-Means & the Gap Statistic

Reference:
Estimating the number of clusters in a data set via the gap statistic
Authors: R. Tibshirani, G. Walther, and T. Hastie
Journal: Journal of the Royal Statistical Society: Series B (Statistical Methodology)
Volume: 63, Pages: 411-423
Year: 2001
DOI: 10.1111/1467-9868.00293

"""
from numpy.typing import *
import numpy as np  #Scientific Computational Package
from gapstatistics.gapstatistics import GapStatistics as gap_statistic  #Optimal Number of Clusters 
from libmr import MR as meta_recognition_tools  #Weibull Fitting
from typing import *

class Meros:

    """
    Interface for calibration based on EVT as well as vector revision based
    on Bendale's formal solution to Open-Set recognition along with our own modifications
    outlined in the GitHub repository.

    Currently, the calibration and revision are based on Bendale's OpenMax, though new methods
    will be added in later versions, namely MetaMax.

    """

    def __init__(
            self,
            activations=None,
            n_centroids_array=None,
            class_vector_distances=None,
            verbose=False
    ):
        pass

    def _get_optimal_n_centroids(
            self,
            activations=None
    ):
        pass

    def _get_centroids(
            self,
            activations=None,
            n_centroids_array=None
    ):
        pass

    def _get_distances(
            self,
            activations=None,
            centroids=None,
            n_centroids_array=None
    ):
        pass

    def _get_weibull_models(
            self,
            activations=None,
            n_centroids_array=None,
            centroids=None
    ):
        pass

    def  fit(
            self,
            activations
    ):
        pass

    def revise(
        self,
        test_activations
    ):
        pass

    def infer(
            self,
            test_activations,
            threshold=0
    ):
        pass

    def fit_revise(
           self,
           class_activations,
           targets,
           test_activations 
    ):
        pass

    def fit_infer(
            self,
            class_activations,
            targets,
            test_activations
    ):
        pass
