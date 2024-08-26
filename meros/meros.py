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
from warnings import warn
import numpy as np  # Scientific Computational Package
from gapstatistics.gapstatistics import GapStatistics  # Optimal Number of Clusters
from libmr import MR as meta_recognition_tools  # Weibull Fitting
from sklearn.cluster import KMeans  # K-Means Clustering
from scipy.special import softmax  # Convert Activations into Probabilities


def weibull_pdf(
    x: NDArray[np.float64], shape: float, scale: float, threshold: float
) -> NDArray[np.float64]:
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


def weibull_cdf(
    x: NDArray[np.float64], shape: float, scale: float, threshold: float
) -> NDArray[np.float64]:
    """Computes CDF of Weibull distribution given shape, scale, and threshold parameters.

    Args:
        x (NDArray[np.float64]): value(s) at which to evaluate CDF.
        shape (float): Weibull shape parameter.
        scale (float): Weibull scale parameter.
        threshold (float): Weibull threshold parameter.

    Returns:
        NDArray[np.float64]: Computed CDF value(s).
    """
    return np.where(x > threshold, 1 - np.exp(-(((x - threshold) / scale) ** shape)), 0)


def weibull_median(shape: float, scale: float, threshold: float) -> float:
    """Computes median of (continuous) Weibull distribution given shape, scale, and threshold parameters.

    Args:
        shape (float): Weibull shape parameter.
        scale (float): Weibull scale parameter.
        threshold (float): Weibull threshold parameter.

    Returns:
        NDArray[np.float64]: Median value of the distribution.
    """
    median = scale * ((-np.log(0.5)) ** (1 / shape)) + threshold
    return median


class OpenSet:
    """
    Interface for calibration based on EVT as well as vector revision based
    on Bendale's formal solution to Open-Set recognition along with our own modifications
    outlined in the GitHub repository.

    Currently, the calibration and revision are based on Bendale's OpenMax, though new methods
    will be added in later versions, namely MetaMax.

    """

    def __init__(self, verbose: bool = True):
        """Instance of Meros Object. Performs EVT and MR based calibration using the activations
           of a trained model's training instances (in a closed-set recognition mode) to then revise
           activations at test time and decide compute a rejection probability for each test instance
           based on open-space risk control.

        Args:
            verbose (bool, optional): Determines whether process updates/messages are printed. Defaults to True.
        """
        self.verbose = verbose
        self.centroids = None
        self.weibull_models = None
        self.n_revised_classes = None

    def _reset(self):
        """Resets attributes."""
        self.centroids = None
        self.weibull_models = None
        self.n_revised_classes = None

    def _message(self, message: str, is_warning: bool = False):
        """Prints process update/message if verbose is toggled to True.

        Args:
            message (str): Message to print.
        """
        if is_warning:
            warn(message)
        elif self.verbose:
            print(message)

    def _compute_class_activations_dict(
        self, activations: ArrayLike, targets: Union[ArrayLike, None] = None
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
        class_activations = {}
        for i in range(activations.shape[1]):
            class_activations[i] = []
        for index in range(activations.shape[0]):
            activation = activations[index]
            if targets is not None:
                target = targets[index]
            else:
                target = np.argmax(activation)
            if np.argmax(activation) == target:
                class_activations[target].append(activation)
        for key, val in class_activations.items():
            class_activations[key] = np.array(val)
        return class_activations

    def _compute_n_centroids_class(
        self, activations: NDArray[np.float64], method: str, max_clusters: int
    ) -> int:
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

        if method == "gapstat":
            max_clusters = min([max_clusters, activations.shape[0]])
            if max_clusters == 0:
                return 0
            gap_statistic_optimizer = GapStatistics(pca_sampling=False)
            n_clusters_opt = gap_statistic_optimizer.fit_predict(
                max_clusters, activations
            )

        else:
            raise ValueError(
                'Unsupported method n_clusters. Use "gapstat" or "mav" or specify cluster numbers (see documentation.)'
            )

        if n_clusters_opt == max_clusters:
            self._message(
                "Optimal cluster number coincides with provided maximum n_clusters.",
                True,
            )
        return n_clusters_opt

    def _compute_optimal_n_centroids_dict(
        self,
        class_activations: Dict[int, NDArray],
        n_clusters: Union[Dict[int, int], str, int],
        max_clusters: int,
    ) -> Dict[int, int]:
        """Compute number of centroids to use for each class and which will be used
           for the distance-to-nearest-centroid approach at Weibull fitting time.

        Args:
            class_activations (Dict[int, NDArray]): Dictionary with targets: [training
                                                    activations]
            n_clusters (Union[Dict[int, int], str, int]): Specify method to use of an
                                                          imposed dictionary of the form
                                                          target: n_centroids. methods are
                                                          strings and support 'mav' for one centroid
                                                          and 'gapstat' for gap statistic clustering.
            max_clusters (int): Number of clusters up to where search for optimal clusters must be performed if
                                looking for optimal number of clusters.

        Raises:
            ValueError: Unspecified or invalid method.

        Returns:
            Dict[int, int]: Dictionary with target:n_centroids.
        """
        n_centroids_dict = {}

        if n_clusters == "mav":
            for key in class_activations.keys():
                n_centroids_dict[key] = 1 if class_activations[key].shape[0] != 0 else 0

        elif isinstance(n_clusters, int):
            for key in class_activations.keys():
                n_centroids_dict[key] = n_clusters
                if n_clusters < 0:
                    raise ValueError("n_clusters is a negative number.")

        elif isinstance(n_clusters, str):
            for key, activations in class_activations.items():
                n_centroids_dict[key] = self._compute_n_centroids_class(
                    activations, n_clusters, max_clusters
                )
        elif isinstance(n_clusters, dict):
            n_centroids_dict = n_clusters
            if sorted([i for i in n_clusters.keys()]) != sorted(
                [j for j in class_activations.keys()]
            ):
                raise ValueError(
                    "Mis-input for n_clusters dictionary. Check target keys."
                )
        else:
            raise ValueError(
                'Unsupported method n_clusters. Use "gapstat" or "mav" or specify cluster numbers (see documentation.)'
            )

        return n_centroids_dict

    def _compute_centroids(
        self, class_activations: Dict[int, NDArray], class_n_centroids: Dict[int, int]
    ):
        """Computes and records centroids using K-Means clustering and, for now,
           exclusively the euclidean distance. Does't return centroids but records them as attribute,

        Args:
            class_activations (Dict[int,NDArray]): Dictionary of the form target: [activations]
            class_n_centroids (Dict[int, int]): Dictionary of the form target: n_centroids
        """
        centroids = {}
        for target, activations in class_activations.items():
            n_clusters = int(class_n_centroids[target])
            if n_clusters == 0:
                centroids[target] = "NO_CENTROIDS"
                continue
            kmeans_clusterer = KMeans(n_clusters, n_init="auto")
            kmeans_clusterer.fit(activations)
            centroids[target] = kmeans_clusterer.cluster_centers_
        self.centroids = centroids

    def _compute_distances(
        self, class_activations: Dict[int, NDArray], centroids: Dict[int, NDArray]
    ) -> Dict[int, NDArray]:
        """Computes distances of class activations from the nearest class centroid.
           The distribution of these distances characterize the penalty received by a vector that
           activates its class according to how its structure compares to the structure of the training
           set, assumed to be a good proxy of the class's ideal representation. See Bendale's OpenMax and
           EVT calibration (docs on https://github.com/sfeilaryan/MeROS .)

        Args:
            class_activations (Dict[int,NDArray]): Dictionary of the form target: [activations]
            centroids (Dict[int,NDArray]): Dictionary of the form target: [class centroids]

        Returns:
            Dict[int,NDArray]: Dictionary of the form target: [training AV distances].
        """
        class_distances = {}
        for target, activations in class_activations.items():
            distances = []
            centers = centroids[target]
            for activation in activations:
                center_distances = [
                    np.linalg.norm(activation - center) for center in centers
                ]
                nearest_center_distance = np.min(center_distances)
                distances.append(nearest_center_distance)
            class_distances[target] = np.array(distances)
        return class_distances

    def _compute_weibull_models(
        self, class_distances: Dict[int, NDArray], weibull_tail_dict: Dict[int, NDArray]
    ):
        """Computes a Weibull fit of the distances and returns the parameters and the median of
           the continuous distribution. See documentation on https://github.com/sfeilaryan/MeROS .
           Weibull is stored - not returned - for revision.

        Args:
            class_distances (Dict[int,NDArray]): Dictionary of the form target: [training AV distances].
                                                (int: array with shape (n_class_instances,))

            weibull_tail_dict (Dict[int,NDArray]):Dictionary of the form target: weibull_parameters.



            In particular, a given class/target's weibull parameters array looks like: (see libmr docs.)

            [scale, shape, threshold sign, threshold absolute value, median]

        """
        class_weibull_parameters = {}
        for target, distances in class_distances.items():
            if distances.shape[0] == 0:
                class_weibull_parameters[target] = "000"
                self._message(
                    f"No class instances for target {target}. Class will not be revised.",
                    True,
                )
                continue
            weibull_tail = weibull_tail_dict[target]
            mr_object = meta_recognition_tools()
            mr_object.fit_high(distances, weibull_tail)
            wb_model = np.array([param for param in mr_object.get_params()])
            wb_median = weibull_median(
                wb_model[1], wb_model[0], wb_model[2] * wb_model[3]
            )
            wb_model[4] = wb_median
            class_weibull_parameters[target] = wb_model
        self.weibull_models = class_weibull_parameters

    def _check_activations(
        self,
        activations: Union[
            Dict[int, NDArray],
            List[List[float]],
            NDArray[np.float64],
            List[NDArray[np.float64]],
        ],
        targets: Union[List[float], NDArray[np.float64], None],
    ):
        """Make sure the model is fitted on a well-conditioned training set.

        Args:
            activations (Union[ Dict[int, NDArray], List[List[float]], NDArray[np.float64], List[NDArray[np.float64]], ]): See Meros.fit
            targets (Union[List[float], NDArray[np.float64], None]): See Meros.fit

        Raises:
            ValueError: Activations and/or target input issues.
        """
        ill_conditioned_activations_message = (
            "You must fix one or more of the following issues:\n"
        )
        ill_conditioned_activations_message += "- You specified a dictionary (target: [activations]) but have not specified all targets including empty ones.\n"
        ill_conditioned_activations_message += "- You specified a dictionary (target: [activations]) but the targets are not exactly the indices of each activation/feature array - the list of targets should like 0,1,...,n_features-1 .\n"
        ill_conditioned_activations_message += (
            "- You specified activations that are not ArrayLike.\n"
        )
        ill_conditioned_activations_message += "- You specified the targets but the targets are not exactly the indices of each activation/feature array - the list of targets should like 0,1,...,n_features-1 .\n"
        ill_conditioned_activations_message += "- Target type error."
        if isinstance(activations, dict):
            target_list = np.array(sorted([i for i in activations.keys()]))
            expected_target_list = np.arange(activations[target_list[0]].shape[1])
            if expected_target_list.shape != target_list.shape:
                raise ValueError(ill_conditioned_activations_message)
            else:
                for i in range(expected_target_list.shape[0]):
                    if expected_target_list[i] != target_list[i]:
                        raise ValueError(ill_conditioned_activations_message)
        else:
            try:
                activations_array = np.array(activations)
            except:
                raise ValueError(ill_conditioned_activations_message)
            if targets is not None:
                if np.max(targets) > activations_array.shape[1] or np.min(targets) < 0:
                    raise ValueError(ill_conditioned_activations_message)
                for target in targets:
                    if isinstance(target, int):
                        continue
                    else:
                        raise ValueError(ill_conditioned_activations_message)

    def fit(
        self,
        activations: Union[
            Dict[int, NDArray],
            List[List[float]],
            NDArray[np.float64],
            List[NDArray[np.float64]],
        ],
        targets: Union[List[float], NDArray[np.float64], None] = None,
        n_centers: Union[None, str, NDArray[np.int64]] = None,
        weibull_tail: Union[int, float] = 0.9,
        weibull_tail_isfraction: bool = True,
        n_max_clusters: int = 10,
        n_revised_classes: Union[int, None] = None,
    ):
        """Calibrates the object to the training results; the correctly classified activations vectors are sorted into classes
           and a Weibull model and centroids are computed for each class, so that this calibration can enable test_instance revision
           and solve open-set recognition at test time using this calibration. See docs.

        Args:
            activations (Union[Dict[int,NDArray], List[List[float]], NDArray[np.float64], List[NDArray[np.float64]]]):

            Array of activations of test_instances. Expected shape: (n_training_samples, n_possible_outputs/n_classes)
            This is the output of a closed-set classifier! Input can also directly be a dictionary with keys target/class and
            values the array of corresponding training activations, in which case targets argument is not read.

            targets (Union[ List[float], NDArray[np.float64], None], optional):

            Corresponding targets of training instances if array (n_training_samples, n_possible_outputs/n_classes)
            is provided. This is used to discard misclassified instances when calibrating. If not specified,
            then activations are taken to have argmax the correct class. Defaults to None.

            n_centers (Union[None, str, NDArray[np.int64]], optional):

            Dictionary, integer, or string specifying the number of centroids per class. Supports dictionary
            of the form target:n_centroids, or integer for constant number across classes, or 'mav' for single mean
            vectors, or 'gapstat' for a optimal number of clusters according to the gap statistic. Defaults to None.

            weibull_tail (Union[int, float], optional):

            Number or proportion of distance data points to use when performing Weibull fit. Chooses the largest values
            (see libmr.MR.fit_high and Bendale's OpenMax available on GitHub). Choose depending on dataset size.
            Defaults to 0.9.


            weibull_tail_isfraction (bool, optional):

            Determine whether to treat previous argument as a fraction or number. Defaults to True.

            n_max_clusters (int, optional)

            Maximum clusters to check optimization of cluster number. Defaults to 10.

            n_revised_classes (Union[int, None], optional):

            Number of top activations to revise - with a decreasing effect anyway. See revise methods and
            the documentation on GitHub. Defaults to None, which will cause a revision in all detected targets with revision effect
            on a given est instance decreasing as activation decreases (we sort the classes to revise by activation; see revise.)

        Raises:
            ValueError: If non-integer provided as Weibull tail and not specified as fraction.

        No return, model fitted and ready for open-space risk control at test time. See revise method.
        """
        self._reset()
        self._check_activations(activations, targets)
        if isinstance(activations, dict):
            class_activations = activations
        else:
            if targets is None:
                self._message(
                    "!!! Targets not given. Assuming all activations yield correct classification !!!"
                )
            class_activations = self._compute_class_activations_dict(activations)

        if n_centers is None:
            self._message(
                "No centroid selection method provided. Using MAVs - one centroid per class (see documentation.)"
            )
            n_centers = "mav"

        n_centroids = self._compute_optimal_n_centroids_dict(
            class_activations, n_centers, n_max_clusters
        )
        self._compute_centroids(class_activations, n_centroids)
        distances = self._compute_distances(class_activations, self.centroids)
        if (not (weibull_tail_isfraction)) and (not (isinstance(weibull_tail, int))):
            raise ValueError("Provide fraction argument or whole number")

        weibull_tail_dict = {}
        if weibull_tail_isfraction:
            for target, activations in class_activations.items():
                weibull_tail_dict[target] = int(weibull_tail * activations.shape[0])
        else:
            for target, activations in class_activations.items():
                weibull_tail_dict[target] = int(weibull_tail)
        self._compute_weibull_models(distances, weibull_tail_dict)
        if n_revised_classes is None:
            self._message(
                "No specified number of top activations to revise; calibrated to revise all activations with decreasing effect. See documentation."
            )
            self.n_revised_classes = class_activations[0].shape[1]
        else:
            self.n_revised_classes = n_revised_classes

    def _revise_vector(self, test_av: NDArray[np.float64]) -> NDArray[np.float64]:
        """Perform OpenMax activation revision and compute an activation for the class of unknown unknowns.
           Called to revise one test_instance. Uses calibration model and parameters.

        Args:
            test_av (NDArray[np.float64]): Test activations (component indices corresponding to targets.)

        Returns:
            NDArray[np.float64]: Revised vector, has shape (input_shape[0]+1,)
        """
        sorted_indices = np.argsort(test_av)[::-1]
        revised_av = np.zeros(test_av.shape[0] + 1)
        for j in range(self.n_revised_classes):
            revised_av[j] = test_av[j]
        unknown_activation = 0
        for i in range(self.n_revised_classes):
            target = sorted_indices[i]
            if isinstance(self.weibull_models[target], str):
                continue
            shape = self.weibull_models[target][1]
            scale = self.weibull_models[target][0]
            shift = self.weibull_models[target][2] * self.weibull_models[target][3]
            median = self.weibull_models[target][4]

            distance = np.min(
                [
                    np.linalg.norm(test_av - centroid)
                    for centroid in self.centroids[target]
                ]
            )

            evaluated_cdf = weibull_cdf(distance, shape, scale, shift)
            shift_from_median = distance - median
            opposite_cdf = weibull_cdf(median - shift_from_median, shape, scale, shift)
            revision_coefficient = (np.abs(evaluated_cdf - opposite_cdf)) * (
                1 - i / self.n_revised_classes
            )
            revised_av[target] *= 1 - revision_coefficient
            unknown_activation += test_av[target] * (revision_coefficient)

        revised_av[-1] = unknown_activation
        return revised_av

    def revise(
        self,
        test_activations: Union[List[float], NDArray[np.float64]],
        softmaxed: bool = False,
    ) -> NDArray[np.float64]:
        """Revises of an array of test activations of shape (n_test_instances, n_activations = n_classes).

        Args:
            test_activations (Union[List[float], NDArray[np.float64]]): Array of test activations.
            softmaxed (bool, optional): Determines if activations are converted to probabilities using the softmax function.
                                        Defaults to False.

        Returns:
            NDArray[np.float64]: Revised activation vectors containing an activation for the unknown class as the last activation
            (or probability if softmaxed.)
        """
        self._message(
            "Note: Largest index feature (one more than maximum provided target index at fitting time) allocated for UNKNOWN/REJECTED."
        )
        test_vectors = np.array(test_activations)
        revised_activations = np.zeros(
            (test_vectors.shape[0], test_vectors.shape[1] + 1)
        )
        for index in range(test_vectors.shape[0]):
            revised_activations[index] = self._revise_vector(test_vectors[index])
        if softmaxed:
            revised_activations = softmax(revised_activations, axis=1)
        return revised_activations

    def infer(
        self,
        test_activations: Union[List[float], NDArray[np.float64]],
        threshold: float = 0.0,
    ) -> NDArray[np.int64]:
        """Returns an array of inferences by revising the vectors and selecting the class with the highest probability.
           Vectors are rejected if they satisfy any of the two criteria:

           1. Highest Probability is that of class unknown (target = -1).
           2. Highest probability is too low (below threshold)

        Args:
            test_activations (Union[List[float], NDArray[np.float64]]): Array of test activations (n_instances, n-activations=n_classes)
            threshold (float, optional): Reject instances as unknown below this probability. Defaults to 0.0.

        Returns:
            NDArray[np.int64]: array of shape (n_input_instances,) with inferred targets for test instances.
        """
        self._message(
            f"Using rejection probability threshold : {threshold}. Note that -1 is UNKNOWN/REJECTED."
        )
        max_known_target = np.max([i for i in self.weibull_models.keys()])
        inferences = np.zeros(test_activations.shape[0])
        revised_probabilities = self.revise(test_activations, True)
        for i in range(revised_probabilities.shape[0]):
            inference = np.argmax(revised_probabilities[i])
            confidence = revised_probabilities[i][inference]
            if confidence < threshold or inference > max_known_target:
                inference = -1
            inferences[i] = inference
        return inferences

    def fit_revise(
        self,
        activations: Union[
            Dict[int, NDArray],
            List[List[float]],
            NDArray[np.float64],
            List[NDArray[np.float64]],
        ],
        targets: Union[List[float], NDArray[np.float64], None] = None,
        n_centers: Union[None, str, NDArray[np.int64]] = None,
        weibull_tail: Union[int, float] = 0.9,
        weibull_tail_isfraction: bool = True,
        n_max_clusters: int = 10,
        n_revised_classes: Union[int, None] = None,
        test_activations: Union[List[float], NDArray[np.float64]] = None,
        softmaxed: bool = False,
    ) -> NDArray[np.float64]:
        """Equivalent to Meros.fit().revise(). See documentation of fit and revise methods."""
        if test_activations is None:
            raise ValueError("Please provide test array!")
        else:
            self.fit(
                activations,
                targets,
                n_centers,
                weibull_tail,
                weibull_tail_isfraction,
                n_max_clusters,
                n_revised_classes,
            )
            return self.revise(test_activations, softmaxed)

    def fit_infer(
        self,
        activations: Union[
            Dict[int, NDArray],
            List[List[float]],
            NDArray[np.float64],
            List[NDArray[np.float64]],
        ],
        targets: Union[List[float], NDArray[np.float64], None] = None,
        n_centers: Union[None, str, NDArray[np.int64]] = None,
        weibull_tail: Union[int, float] = 0.9,
        weibull_tail_isfraction: bool = True,
        n_max_clusters: int = 10,
        n_revised_classes: Union[int, None] = None,
        test_activations: Union[List[float], NDArray[np.float64]] = None,
        threshold: float = 0.0,
    ) -> NDArray[np.int64]:
        """Equivalent to Meros.fit().infer(). See documentation of fit and infer methods."""
        if test_activations is None:
            raise ValueError("Please provide test array!")
        else:
            self.fit(
                activations,
                targets,
                n_centers,
                weibull_tail,
                weibull_tail_isfraction,
                n_max_clusters,
                n_revised_classes,
            )
            return self.infer(test_activations, threshold)

        return

    def get_centroids(self) -> Dict[int, NDArray[np.float64]]:
        """Getter for centroids attributes.

        Returns:
            Dict[int, NDArray[np.float64]]: Dictionary of form target: [array of centroids]
        """
        if self.centroids is None:
            self._message(
                "Attribute not assigned yet. Fit (or fit_revise) the wrapper first!"
            )
        return self.centroids

    def get_weibull_models(self) -> Dict[int, NDArray[np.float64]]:
        """Getter for fitted Weibull parameters.

        Returns:
            Dict[int, NDArray[np.float64]]: Dictionary of form target: weibull parameters. See _compute_weibull_models method doc.
        """
        if self.weibull_models is None:
            self._message(
                "Attribute not assigned yet. Fit (or fit_revise) the wrapper first!"
            )
        return self.weibull_models
