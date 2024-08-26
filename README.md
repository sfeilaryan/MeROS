# MeROS - Meta Recognition & Open Set Classification by Deep Models
## Synopsis
This python package essentially consists of a compact wrapper for meta-recognition inspired functionalities that all revolve around Bendale and Boult's [[1]](#1) <i> Toward Open Set Deep Networks</i> where they outline a <b>calibration</b> after training and a <b>revision</b> at test time to reject unknown events by minimizing open-space risk, making the model more robust to fooling and adversarial events than a known-class-activation-softmax procedure with a confidence threshold for rejection of unknowns. Readers and MeROS users that are not familiar with OpenMax are encouraged to check Bendale and Boult's paper out [[1]](#1) and to subsequently quick this document a quick read as we integrate some enhancements and optimizations that may be mentioned in the aforementioned paper for discussion purposes but that weren't relevant enough to the research to be implemented or elaborated.

## Installation

Available on PyPI with very weak dependencies (check out `requirements.txt` for now but basic packages, the versions of which have only been pinned because they have been tested on these versions but the accessed objects are most probably available on most versions of these packages; see documentation.)

```
pip install meros
```
### Important

Please make sure you have the Python development headers from `python-dev` installed <b>system-wide</b>. On macOS, Homebrew ensures them. You will notice they are missing if you encounter an error as you try to install `libMR`, which requires an API for C extension. You can install these headers on Linux using:
```
sudo apt-get install python3-dev
```

Instead of `python-3`, you can specify your Python version in `X.X` form if you have several Python versions to power different (virtual) environments. Once yo have these headers, you should be fine on the supported Python versions (3.8-3.9 for now.)

## Motivation & Preliminary Information
Typical neural networks used for classification tasks and problems are limited to closed-set classification by nature. The typical output activations resultant from a test input are usually converted to a discrete probability distribution reflecting the model's relative confidences exclusively over the classes or targets it has encountered at training time. Hence, feeding the model an input that doesn't correspond to any of the target classes still results in confidences over the aforementioned target classes that add up to 1 as a result of the softmax operation applied on the model's penultimate layer. In a lot of real-world deployment settings, our models can and will encounter new behavior it has not encountered at training time and should know to detect/reject/signal these events for designer investigation and a possible reconsideration of the model's training. This is the heart of open-set classification and recognition. 

A natural reaction to such a prompt is to invoke the heuristic that unknown instances characterize themselves by eccentricity, leading to low activations across the classes and low maximal confidences, in which a threshold on probability would suffice to solve the problem. Unfortunately, research shows that "fooling" instances of unknown classes can easily be generated to yield high activations in certain classes. [[4]](#4) For this reason, Bendale's approach relies on discarding the softmax layer which is far too connected to a  "closedness" of the model and adapt meta-recognition concepts and control of the open-space risk in the feature space of the model's penultimate layer. This process is described below and properly elaborated in Bendale's paper. The central idea is that by constructing a calibration of the expected structure of classes as described by the penultimate activation layer can give us an idea of how eccentric test inputs are based on how much they resemble the representative vectors of the classes they activate the most, ultimately enabling us to estimate the risk associated with making a classification in one of the known classes. The failure of the softmax equipped with a threshold is due to the fact that low activations result in high probabilities provided the the other activations remain substantially lower; this is obvious from the normalization involved in the softmax application. More precisely, a formal solution to the open-set recognition problem must satisfy the compact abating probability model. [[1]](#1)

## EVT and Meta-Recognition Based Calibration
The first main functionality this package enables is the extreme-value theory and meta-recognition based calibration of our system. Once a deep network is trained, the non-softmaxed activations of the training instances are placed in separate piles depending on the class they correspond to, discarding incorrectly classified instances (in which the target class's activation isn't the highest.) We then compute, for each class, a representative vector. In Bendale's paper, an MAV - mean activation vector computed by the vector  components of each class's activations - is computed for each class. An optimization where one uses several centroids is discussed, and this package supports both an MAV approach, a multiple centroid approach based on a K-Means clustering for each class with K selected as the optimal number of clusters according the gap statistic [[3]](#3) which was motivated by a formalization of the heuristic behind the well-known "elbow method." We only support the euclidean distance clustering scheme for now and look forward to implementing more centroid choices such as a choice based on a maximization of the silhouette coefficient. The number of centroids can also be explicitly specified for all the classes using a dictionary with target classes (integers) as keys and the number of centroids for each class as values when calling the `Meros.fit` method. Once the centroids are computed and stored, we compute, for each class, the distance between the activation vectors of each class and the nearest class centroid. We end up, for a given class, with an array of shape `(n_training_instances,)` of distances that we then fit with a Weibull distribution (see Algorithm 1 here [[1]](#1)) and record the parameters - or simply the Weibull model - for each class. We will use this fit the judge how eccentric a test vector is when it activates this class by comparing to the class representative(s) and judging its distance using the training distances, assumed to be a good enough proxy of the class's true structure. We also record each Weibull model's median and this is explained in the next section.

## Revision at Test Time of Activations
The second main functionality of supported by this package is the actual discrimination/revision at test time based on open-space risk rather than a naive thresholding scheme. 

### IMPORTANT: Due to apparent inconsistencies/errors/typos in the currently available versions of [[1]](#1), the revision scheme sligthly differs from Algorithm 2 in Bendale and Boult's paper and we elaborate this choice here.

Consider a give test instance that a deep network must predict the probabilities of infer a target for. As we've done for the training, we first consider the non-normalized activations forming the test activation vector. We consider the top $\alpha$ classes sorted in descending order of activation - this is a hyperparameter selected at fitting time. The idea is that we only want to revise classes with high activation susceptible to being the target and hence requiring a comparison against the calibrated Weibull model. We initialize the unknown activation $t_0=0$ and start with the most activated class. We compute the test AV's distance to the target class's nearest centroid and evaluate the class's Weibull CDF at this distance. We make the following observations:

- A large CDF evaluation means that this vector is relatively far from the representatives of the considered target class and should be punished for activating it as much as it did.

- Similarly, a low CDF means that this vector is relatively close to the class representative(s).

- This is only relevant for the well-activated classes, because instances presumed to be of a certain class are not expected to look like the representatives of another class, much less so when the latter is a class much lower in the list of classes sorted by descending activation.

- Lower distance doesn't necessarily mean better; the mere choice of a Weibull fit and the performance of said distribution imply the possibillity of a characteristic distance between class vectors and their nearest centroid, such as for circular clusters in two dimensions.

These motivations motivate the following procedure. When revising the $i$-th most activated class, we compute the Weibull CDF at the distance of the test AV from the nearest class centroid, as well the CDF at the symmetric of this distance about the median of the distribution. We then compute the difference between these values; this is the integral of the PDF between the computed distance of the point on the other side of the median that is at the same difference from the median (of median is 2 and distance is 1, the symmetric point in question is 3.) We record this coefficient and multiply by $\frac{\alpha - i}{\alpha}$ and record it as $\beta_i$. We then take away (subtract) a proportion $\beta_i$ from this class's activation and add it to the unknown activation. We do this for the $\alpha$ most activated classes. We outline the meaning of each step below:

- **The integral of the PDF over the region between distance and symmetric value about the media:** this measures how far away from the median the value is, regardless of direction (to accommodate the 4th bullet point in the above list) and is a good measure of how likely it is for the computed distance to be an outlier. Sampling the median distance results in a 0 coefficient or no addition to unknown uncertainty, whereas huge and tiny CDF's yield nearly maximal activation subtraction.

- **We record this coefficient and multiply by $\frac{\alpha - i}{\alpha}$:** this pertains to the second bullet point in the above list and weights the revising effect by the class's rank, so that only the highest activations are strongly revised. This could have been an exponent and the fact that it is a coefficient is inspired from the methodology outlined in [[1]](#1).

We end up with an unknown activation which we append to the AV, which is then softmaxed. The instance is deemed unknown if either the unknown class has the highest probability or if the highest probability is below a preset threshold. Thus, to beat OpenMax, a test AV has to avoid activating the unknown class by conforming to the structure of activated classes individually and also has to avoid activating several classes or the renormalization will bring down the probability below the threshold. In SoftMax it is not required to actually highly activate the fooling target as long as the relative activations of the other classes are substantially lower to bypass the threshold and rejection mechanism. Revision is done with the `revise` and `infer` methods, where the latter applies the rejection criteria.
## Citation
If you use this software in your research, please cite it as follows:

```bibtex
@software{sfeila_ryan_2024,
  author = {Ryan Sfeila},
  title = {MeROS - Meta-Recognition Tools for Open-Set Classification},
  version = {0.0.3},
  year = {2024},
  url = {https://github.com/sfeilaryan/MeROS},
  license = {BSD-3-Clause}
}
```


## Important Notes: Deviations from Bendale and Boult's Paper
It is important to note that the revision schemes discussed above as well as the multiple centroid approach are not implemented in the paper that has directed most of what this package implements. It's been empirically verified by the author on an audio recognition case that these new implementations do enhance the mechanism. The reader is invited to contact the author for more details regarding this case.
## Upcoming Updates
### Documentation
- Demonstrations and Tutorials
- Realistic Example Implementation Using a Basic Architecture Powered by Keras.

### Developments
- Quick Calibration Visualization Methods (Weibull fit quality and resultant centroids.)
- More Convenient Wrapping Functionality by Focusing on Various Deep Network Architectures
- MetaMax [[5]](#5). Based on non-match scores rather vectorial structures.

### Ambitious Plans
- Try to enable OpenMax as a layer option in `tensorflow.keras` the same way it is done for softmax. Extra considerations on calibration required, as well as contact with the TF team.

## Contributions

Ryan Sfeila has been in charge of deploying the current available version of `Meros` and documenting its structure and functionality.

## References
<a id="1">[1]</a> 
Bendale, Abhijit, and Terrance Boult. "Towards Open Set Deep Networks." *Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on*, 2016, IEEE.

<a id="2">[2]</a> Scheirer, Walter J., Anderson Rocha, Ross Michaels, and Terrance E. Boult. "Meta-Recognition: The Theory and Practice of Recognition Score Analysis." *IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)*, vol. 33, no. 8, 2011, pp. 1689-1695.

<a id="3">[3]</a> Tibshirani, Robert, Guenther Walther, and Trevor Hastie. "Estimating the Number of Clusters in a Data Set via the Gap Statistic." *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, vol. 63, 2001, pp. 411-423. doi:10.1111/1467-9868.00293.

<a id="4">[4]</a> A. Nguyen, J. Yosinski, and J. Clune. Deep neural networks
are easily fooled: High confidence predictions for unrecognizable images. In Computer Vision and Pattern Recognition
(CVPR), 2015 IEEE Conference on. IEEE, 2015.

<a id="5">[5]</a> Lyu, Zongyao, et al. "MetaMax: Improved Open-Set Deep Neural Networks via Weibull Calibration." arXiv, 2022.
