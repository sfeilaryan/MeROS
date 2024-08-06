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


def weibull_pdf(x: NDArray[np.float64], shape: float, scale: float, threshold: float) -> NDArray[np.float64]:
    """Computes PDF of Weibull distribution given shape, scale, and threshold parameters.

    Args:
        x (NDArray[np.float64]): value(s) at which to evaluate PDF.
        shape (float): Weibull shape parameter.
        scale (float): Weibull scale parameter. 
        threshold (float): Weibull threshold parameter.

    Returns:
        NDArray[np.float64]: Computed PDF value(s).
    """
    return np.where(
        x > threshold,
        (shape / scale)
        * ((x - threshold) / scale) ** (shape - 1)
        * np.exp(-(((x - threshold) / scale) ** shape)),
        0,
    )

def weibull_cdf(x: NDArray[np.float64], shape:float, scale:float, threshold:float)-> NDArray[np.float64]:
    """Computes CDF of Weibull distribution given shape, scale, and threshold parameters.

    Args:
        x (NDArray[np.float64]): value(s) at which to evaluate CDF.
        shape (float): Weibull shape parameter.
        scale (float): Weibull scale parameter. 
        threshold (float): Weibull threshold parameter.

    Returns:
        NDArray[np.float64]: Computed CDF value(s).
    """
    return np.where(
        x > threshold,
        1
        - np.exp(-(((x - threshold) / scale) ** shape)),
        0
    )

def weibull_median(shape:float, scale:float, threshold:float)->float:
    """Computes median of (continuous) Weibull distribution given shape, scale, and threshold parameters.

    Args:
        shape (float): Weibull shape parameter.
        scale (float): Weibull scale parameter. 
        threshold (float): Weibull threshold parameter.

    Returns:
        NDArray[np.float64]: Median value of the distribution. 
    """
    median = scale*((-np.log(0.5))**(1/shape)) + threshold
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
            verbose:bool=True
    ):
        """Instance of Meros Object. Performs EVT and MR based calibration using the activations
           of a trained model's training instances (in a closed-set recognition mode) to then revise
           activations at test time and decide compute a rejection probability for each test instance
           based on open-space risk control.

        Args:
            verbose (bool, optional): Determines whether process updates/messages are printed. Defaults to True.
        """
        self.verbose=verbose
        self.centroids=None
        self.weibull_models=None
        self.n_revised_classes=None

    def _reset(self):
        """Resets attributes.
        """
        self.centroids=None
        self.weibull_models=None
        self.n_revised_classes=None

    def _message(
            self,
            message:str
    ):
        """Prints process update/message if verbose is toggled to True.

        Args:
            message (str): Message to print.
        """
        if self.verbose:
            print(message)
    
    def _compute_class_activations_dict(
            self,
            activations:ArrayLike,
            targets:Union[ArrayLike, None]=None
    ) -> Dict[int, NDArray[np.float64]]:
        """Turns an array of activations (presumably some ModelObject.predict output on
           the training set) and creates a dictionary with targets for keys and the 
           activations of the (correctly classified) training instances of a the given targets
           in an array as values. If targets are specified (and they are expected to be
           integers numbered from 0 to n_classes-1 so that they can be used an indices)
           then we make sure to ignore incorrectly classified instances. If targets is None
           we assume all arrays have for target that for which they have the highest activation.
           Returns the dictionary as it is useless to store it as an attribute (with meros's current
           functionality.)

        Args:
            activations (ArrayLike): Two-dimensional array of shape (n_samples, n_classes)
                                    corresponding to training instance activations.

            targets (Union[ArrayLike, None], optional): Corresponding targets. If not 
                                                        specified then assumed argmax
                                                        of activations.
                                                        Defaults to None.

        Returns:
            Dict[int, NDArray[np.float64]]: Dictionary with target keys and activations array
                                            as values.
        """
        activations = np.array(activations)
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
        return class_activations


    def _compute_n_centroids_class(
            self,
            activations: NDArray[np.float64],
            method: str,
            max_clusters: int
    )-> int:
        """Given an array of activations, returns the optimal number of clusters to
           use to represent the array based on passed criterion (supports Gap Statistic for now)
           (see documentation.) Based on K-Means with euclidean distance for now.

           Warning message printed if algorithm opts for maximum number of clusters (would advise pushing
           the number back or forcing something based on a heuristic.)

        Args:
            activations (NDArray[np.float64]): Activations (of a given class) to cluster.
            method (str):Criterion to choose optimal number of centroids.
                         For now supports gap statistic and will include silhouette coefficient.
            max_clusters (int): Maximum number of clusters to check performance for.

        Raises:
            ValueError: If an invalid criterion string method is passed.

        Returns:
            int: Optimal number of centroids/clusters.
        """
        
        if method=='gapstat':
            gap_statistic_optimizer = GapStatistics(pca_sampling=False)
            n_clusters_opt= gap_statistic_optimizer.fit_predict(max_clusters, activations)

        else:
            raise ValueError('Unsupported method n_clusters. Use "gapstat" or "mav" or specify cluster numbers (see documentation.)')
        
        if n_clusters_opt==max_clusters:
            self._message('WARNING: Optimal cluster number coincides with provided maximum n_clusters.')
        return n_clusters_opt
        

    def _compute_optimal_n_centroids_dict(
            self,
            class_activations: Dict[int, NDArray],
            n_clusters: Union[Dict[int, int], str, int],
            max_clusters: int
    )-> Dict[int, int]:
        
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
     
        return n_centroids_dict

            

    def _compute_centroids(
            self,
            class_activations: Dict[int,NDArray],
            class_n_centroids:Dict[int, int]
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
            class_activations:Dict[int,NDArray],
            centroids: Dict[int,NDArray]
    )->Dict[int,NDArray] :
        class_distances={}
        for target, activations in class_activations.items():
            distances=[]
            centers=centroids[target]
            for activation in activations:
                center_distances = [np.linalg.norm(activation-center) for center in centers]
                nearest_center_distance= np.min(center_distances)
                distances.append(nearest_center_distance)
            class_distances[target] = distances
        return class_distances

    def _compute_weibull_models(
            self,
            class_distances:Dict[int,NDArray],
            weibull_tail_dict:Dict[int,NDArray]
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
            activations:Union[Dict[int,NDArray], List[List[float]], NDArray[np.float64], List[NDArray[np.float64]]],
            targets: Union[ List[float], NDArray[np.float64], None] =None,
            n_centers:Union[None, str, NDArray[np.int64]] =None,
            weibull_tail:Union[int, float]=0.9,
            weibull_tail_isfraction:bool=True,
            n_max_clusters:int=10,
            n_revised_classes:Union[int, None] = None
    ):
        if isinstance(activations, dict):
            class_activations=np.array(activations)
        else:
            if targets is None:
                self._message('Targets not given. Assuming all activations yield correct classification.')
            class_activations=self._compute_class_activations_dict(activations)
        
        if n_centers is None:
            self._message('No centroid selection method provided. Using MAVs - one centroid per class (see documentation.)')
            n_centers='mav'

        n_centroids = self._compute_optimal_n_centroids_dict(class_activations,
                                                                  n_centers,
                                                                  n_max_clusters)
        self._compute_centroids(class_activations, n_centroids)
        distances =self._compute_distances(self.class_activations, self.centroids)
        if (not(weibull_tail_isfraction)) and (not(isinstance(weibull_tail, int))):
            raise ValueError('Provide fraction argument or whole number')
        
        weibull_tail_dict = {}
        if weibull_tail_isfraction:
            for target, activations in class_activations:
                weibull_tail_dict[target] = int(weibull_tail*activations.shape[0])
        else:
            for target, activations in class_activations:
                weibull_tail_dict[target] = int(weibull_tail)
        self._compute_weibull_models(
            distances, 
            weibull_tail_dict
        )
        if n_revised_classes is None:
            self._message('No specified number of top activations to revise; calibrated to revise all activations with decreasing effect.')
            self.n_revised_classes = max([i for i in self.class_activations.keys()])

    def _revise_vector(
            self,
            test_av: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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
        test_activations:Union[List[float], NDArray[np.float64]],
        softmaxed:bool=False
    )->NDArray[np.float64]:
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
            test_activations:Union[List[float], NDArray[np.float64]],
            threshold:float=0.0
    )->NDArray[np.int64]:
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
            activations:Union[Dict[int,NDArray], List[List[float]], NDArray[np.float64], List[NDArray[np.float64]]],
            targets: Union[ List[float], NDArray[np.float64], None] =None,
            n_centers:Union[None, str, NDArray[np.int64]] =None,
            weibull_tail:Union[int, float]=0.9,
            weibull_tail_isfraction:bool=True,
            n_max_clusters:int=10,
            n_revised_classes:Union[int, None] = None,
            test_activations:Union[List[float], NDArray[np.float64]] = None,
            softmaxed:bool = False
    )->NDArray[np.float64]:
        if test_activations is None:
            raise ValueError('Please provide test array!')
        else:
            self.fit(activations,targets,n_centers,weibull_tail, weibull_tail_isfraction,
                            n_max_clusters, n_revised_classes)
            return self.revise(test_activations, softmaxed)

    def fit_infer(
            self,
            activations:Union[Dict[int,NDArray], List[List[float]], NDArray[np.float64], List[NDArray[np.float64]]],
            targets: Union[ List[float], NDArray[np.float64], None] =None,
            n_centers:Union[None, str, NDArray[np.int64]] =None,
            weibull_tail:Union[int, float]=0.9,
            weibull_tail_isfraction:bool=True,
            n_max_clusters:int=10,
            n_revised_classes:Union[int, None] = None,
            test_activations:Union[List[float], NDArray[np.float64]] = None,
            threshold:float = 0.0
    )->NDArray[np.int64]:
        if test_activations is None:
            raise ValueError('Please provide test array!')
        else:
            self.fit(activations,targets,n_centers,weibull_tail, weibull_tail_isfraction,
                            n_max_clusters, n_revised_classes)
            return self.infer(test_activations, threshold)

        return

    def get_centroids(self)-> Dict[int, NDArray[np.float64]]:
        if self.centroids is None:
            self._message('Attribute not assigned yet. Fit (or fit_revise) the wrapper first!')
        return self.centroids

    def get_weibull_models(self)-> Dict[int, NDArray[np.float64]]:
        if self.weibull_models is None:
            self._message('Attribute not assigned yet. Fit (or fit_revise) the wrapper first!')
        return self.weibull_models