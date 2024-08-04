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
from typing import *
from numpy.typing import *
import numpy as np  #Scientific Computational Package
from gapstatistics.gapstatistics import GapStatistics  #Optimal Number of Clusters 
from libmr import MR as meta_recognition_tools  #Weibull Fitting
from sklearn.cluster import KMeans  #K-Means Clustering
from scipy.special import softmax   #Convert Activations into Probabilities


def weibull_pdf(x, shape, scale, threshold):
    return np.where(
        x > threshold,
        (shape / scale)
        * ((x - threshold) / scale) ** (shape - 1)
        * np.exp(-(((x - threshold) / scale) ** shape)),
        0,
    )

def weibull_cdf(x, shape, scale, threshold):
    return np.where(
        x > threshold,
        1
        - np.exp(-(((x - threshold) / scale) ** shape)),
        0
    )

def weibull_median(shape, scale, threshold):
    x = np.linspace(threshold, threshold + 5 * scale, 1000)
    y = weibull_cdf(x, shape, scale, threshold)
    median=0
    for j in range(y.shape[0]):
        if y[j] >= 0.5:
            median = x[j]
            break
    return median


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
            verbose=True
    ):
        self.verbose=verbose
        self.class_activations=None
        self.n_centroids=None
        self.centroids=None
        self.nearest_centroid_distances=None
        self.max_clusters=None
        self.weibull_tail_dict=None
        self.weibull_models=None
        self.n_revised_classes=None
        self.inference_threshold=0.0

    def _reset(self):
        self.class_activations=None
        self.n_centroids=None
        self.centroids=None
        self.nearest_centroid_distances=None
        self.weibull_tail_dict=None
        self.weibull_models=None
        self.n_revised_classes=None
        self.inference_threshold=0.0

    def _message(
            self,
            message
    ):
        if self.verbose:
            print(message)
    
    def _compute_class_activations_dict(
            self,
            activations,
            targets=None
    ):
        class_activations={}
        for index in range(activations.shape[0]):
            if targets is not None:
                target=targets[index]
            else:
                target = np.argmax(activation)
            activation=activations[index]
            if np.argmax(activation) != target:
                continue
            elif activation in class_activations:
                class_activations[target]=[activation]
            else:
                class_activations.append(activation)
        for key, val in class_activations.items():
            class_activations[key]=np.array(val)
        self.class_activations=class_activations


    def _compute_n_centroids_class(
            self,
            activations,
            method,
            max_clusters
    ):
        
        if method=='gapstat':
            gap_statistic_optimizer = GapStatistics(pca_sampling=False)
            n_clusters_opt= gap_statistic_optimizer.fit_predict(max_clusters, activations)

        else:
            raise ValueError('Unsupported method n_clusters. Use "gapstat" or "mav" or specify cluster numbers (see documentation.)')
        
        return n_clusters_opt
        

    def _compute_optimal_n_centroids_dict(
            self,
            class_activations,
            n_clusters,
            max_clusters
    ):
        
        n_centroids_dict={}

        if n_clusters == 'mav':
            for key in class_activations.keys():
                n_centroids_dict[key] = 1

        elif isinstance(n_clusters, int):
            for key in class_activations.keys():
                n_centroids_dict[key] = n_clusters

        elif isinstance(n_clusters, str):
            for key, activations in class_activations.items():
                n_centroids_dict[key] = self._compute_n_centroids_class(activations,
                                                                        n_clusters,
                                                                        max_clusters)
        elif isinstance(n_clusters,dict):
            n_centroids_dict = n_clusters

        else:
            raise ValueError('Unsupported method n_clusters. Use "gapstat" or "mav" or specify cluster numbers (see documentation.)')
        self.n_centroids=n_centroids_dict

            

    def _compute_centroids(
            self,
            class_activations,
            class_n_centroids
    ):
        
        centroids = {}
        for target, activations in class_activations.items():
            n_clusters = int(class_n_centroids[target])
            kmeans_clusterer=KMeans(n_clusters)
            kmeans_clusterer.fit(activations)
            centroids[target]=kmeans_clusterer.cluster_centers_
        self.centroids=centroids
        

    def _compute_distances(
            self,
            class_activations,
            centroids
    ):
        class_distances={}
        for target, activations in class_activations.items():
            distances=[]
            centers=centroids[target]
            for activation in activations:
                center_distances = [np.linalg.norm(activation-center) for center in centers]
                nearest_center_distance= np.min(center_distances)
                distances.append(nearest_center_distance)
            class_distances[target] = distances
        self.nearest_centroid_distances=class_distances

    def _compute_weibull_models(
            self,
            class_distances,
            weibull_tail_dict
    ):
        class_weibull_parameters = {}
        for target, distances in class_distances.items():
            weibull_tail = weibull_tail_dict[target]
            mr_object=meta_recognition_tools.fit_high(distances, weibull_tail)
            wb_model = mr_object.get_params()
            wb_median = weibull_median(wb_model[1], wb_model[0], wb_model[2]*wb_model[3])
            wb_model[4]=wb_median
            class_weibull_parameters[target]=wb_model
        self.weibull_models = class_weibull_parameters

    def  fit(
            self,
            activations,
            targets=None,
            n_centroids=None,
            weibull_tail=0.9,
            weibull_tail_isfraction=True,
            n_max_clusters=10,
            n_revised_classes = None
    ):
        if isinstance(activations, dict):
            self.class_activations=activations
        else:
            if targets is None:
                self._message('Targets not given. Assuming all activations yield correct classification.')
            self.class_activations=self._compute_class_activations_dict(activations)
        
        if n_centroids is None:
            self._message('No centroid selection method provided. Using MAVs - one centroid per class (see documentation.)')
            n_centroids='mav'

        self.n_centroids = self._compute_optimal_n_centroids_dict(self.class_activations,
                                                                  n_centroids,
                                                                  n_max_clusters)
        self.centroids =self._compute_centroids(self.class_activations, self.n_centroids)
        self.nearest_centroid_distances=self._compute_distances(self.class_activations, self.centroids)
        if (not(weibull_tail_isfraction)) and (not(isinstance(weibull_tail, int))):
            raise ValueError('Provide fraction argument or whole number')
        
        weibull_tail_dict = {}
        if weibull_tail_isfraction:
            for target, activations in self.class_activations:
                weibull_tail_dict[target] = int(weibull_tail*activations.shape[0])
        else:
            for target, activations in self.class_activations:
                weibull_tail_dict[target] = int(weibull_tail)
        self.weibull_tail_dict=weibull_tail_dict
        self.weibull_models=self._compute_weibull_models(
            self.class_activations, 
            self.weibull_tail_dict
        )
        if n_revised_classes is None:
            self.n_revised_classes = max([i for i in self.class_activations.keys()])

    def _revise_vector(
            self,
            test_av
    ):
        sorted_indices = np.argsort(test_av)[::-1]
        revised_av = np.zeros(test_av.shape[0] + 1)
        for j in range(self.n_revised_classes):
            revised_av[j] = test_av[j]
        unknown_activation = 0
        for i in range(self.n_revised_classes):
            target = sorted_indices[i]
            shape = self.weibull_models[target][1]
            scale = self.weibull_models[target][0]
            shift = self.weibull_models[target][2] * self.weibull_models[target][3]
            median = self.weibull_models[target][4]

            distance = np.min([np.linalg.norm(test_av - centroid) for centroid in self.centroids[target]])

            evaluated_cdf = weibull_cdf(distance, shape, scale, shift)
            shift_from_median = distance - median
            opposite_cdf = weibull_cdf(median - shift_from_median, shape, scale, shift)
            revision_coefficient = (np.abs(evaluated_cdf - opposite_cdf))*(1 - i/self.n_revised_classes)
            revised_av[target] *= (1 - revision_coefficient)
            unknown_activation += test_av[target] * (revision_coefficient)

        revised_av[-1] = unknown_activation
        
        return revised_av

    def revise(
        self,
        test_activations,
        softmaxed=False
    ):
        self._message(f'Note that largest index (one more than maximum provided target index at fitting time) is UNKNOWN/REJECTED.')
        test_vectors = np.array(test_activations)
        revised_activations = np.zeros((test_vectors.shape[0],test_vectors.shape[1]+1))
        for index in range(test_vectors.shape[0]):
            revised_activations[index] = self._revise_vector(test_vectors[index])
        if softmaxed:
            revised_activations=softmax(revised_activations, axis=1)
        return revised_activations

    def infer(
            self,
            test_activations,
            threshold=0
    ):
        self._message(f'Using rejection probability threshold : {threshold}. Note that -1 is UNKNOWN/REJECTED.')
        max_known_target = np.max([i for i in self.weibull_models.keys()])
        inferences = np.zeros()
        revised_probabilities=self.revise(test_activations, True)
        for i in range(revised_probabilities.shape[0]):
            inference = np.argmax(revised_probabilities[i])
            confidence = revised_probabilities[i][inference]
            if confidence<threshold or inference> max_known_target:
                inference=-1
            inferences[i] = inference
        return inferences


    def fit_revise(
            self,
            activations,
            targets=None,
            n_centroids=None,
            weibull_tail=0.9,
            weibull_tail_isfraction=True,
            n_max_clusters=10,
            n_revised_classes = None,
            test_activations = None,
            softmaxed = False
    ):
        if test_activations is None:
            raise ValueError('Please provide test array!')
        else:
            self.fit(activations,targets,n_centroids,weibull_tail, weibull_tail_isfraction,
                            n_max_clusters, n_revised_classes)
            return self.revise(test_activations, softmaxed)

    def fit_infer(
            self,
            activations,
            targets=None,
            n_centroids=None,
            weibull_tail=0.9,
            weibull_tail_isfraction=True,
            n_max_clusters=10,
            n_revised_classes = None,
            test_activations = None,
            threshold = 0
    ):
        if test_activations is None:
            raise ValueError('Please provide test array!')
        else:
            self.fit(activations,targets,n_centroids,weibull_tail, weibull_tail_isfraction,
                            n_max_clusters, n_revised_classes)
            return self.infer(test_activations, threshold)

        return

    def get_class_n_centroids(self):
        return self.n_centroids

    def get_centroids(self):
        return self.centroids

    def get_distances(self):
        return self.nearest_centroid_distances

    def get_weibull_models(self):
        return self.weibull_models