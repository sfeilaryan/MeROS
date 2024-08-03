"""
A compact wrapper for meta-recognition and more specifically open-set classification functionality.
The main implementations are summarized below along with their respective references if applicable.

- Model Meta-Recognition Calibration Based on EVT - Centroid(s) and Weibull Distance Models - Tools



- Computing at Run Time Probability of Unseen Unknown - Reject Novelty - Open-Space Risk Control - OpenMax

Reference:
Towards Open Set Deep Networks
Authors: Abhijit Bendale and Terrance Boult
Presented at: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
Published by: IEEE
"""

import numpy as np
from gapstatistics.gapstatistics import GapStatistics as gap_statistic
from libmr import MR as meta_recognition_tools



