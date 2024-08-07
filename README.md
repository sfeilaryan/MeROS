# MeROS - Meta Recognition for Open Set Classification by Deep Models
## Synopsis
This python package essentially consists of a compact wrapper for meta-recognition inspired functionalities that all revolve around Bendale and Boult's [[1]](#1) <i> Toward Open Set Deep Networks</i> where they outline a <b>calibration</b> after training and a <b>revision</b> at test time to reject unknown events by minimizing open-space risk, making the model more robust to fooling and adversarial events than a softmax+threshold procedure.
## Motivation

## EVT and Meta-Recognition Based Calibration

## Revision at Test Time of Activations

## Rejection of Unknowns

## References
<a id="1">[1]</a> 
Bendale, Abhijit, and Terrance Boult. "Towards Open Set Deep Networks." *Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on*, 2016, IEEE.

<a id="1">[2]</a> Scheirer, Walter J., Anderson Rocha, Ross Michaels, and Terrance E. Boult. "Meta-Recognition: The Theory and Practice of Recognition Score Analysis." *IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)*, vol. 33, no. 8, 2011, pp. 1689-1695.

<a id="1">[3]</a> Tibshirani, Robert, Guenther Walther, and Trevor Hastie. "Estimating the Number of Clusters in a Data Set via the Gap Statistic." *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, vol. 63, 2001, pp. 411-423. doi:10.1111/1467-9868.00293.
