"""
Various feature extraction methods.

Authors: Carsten Tusk, Kostas Stamatiou
Contact: kostas.stamatiou@digitalglobe.com
"""

import numpy as np


def vanilla_features(data):
    """The simplest feature extractor.

       Args:
           data (numpy array): Pixel data vector.

       Yields:
           A vector with the mean, std and variance of data.
    """
    
    yield [ np.mean(data), np.std(data), np.var(data), np.min(output_data) ]
    

def water_features(data):
	"""Feature extractor for water detection (e.g., swimming pools over land)

       Args:
           data (numpy array): Pixel data vector.

       Yields:
           Feature vector (numpy array).
    """

    output_data = sa.spectral_angles(data)

    band28_ratio = (data[1,:,:]-data[7,:,:])/(data[1,:,:]+data[7,:,:])
    
    band37_ratio = (data[2,:,:]-data[6,:,:])/(data[2,:,:]+data[6,:,:])

    yield [ band28_ratio.max(), band37_ratio.max(), np.min(output_data) ]
