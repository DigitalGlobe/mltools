"""
Various feature extraction methods.

Authors: Carsten Tusk, Kostas Stamatiou
Contact: kostas.stamatiou@digitalglobe.com
"""

import numpy as np
import spectral_angle as sa


def vanilla_features(data):
    """The simplest feature extractor.

       Args:
           data (numpy array): Pixel data vector.

       Yields:
           A vector with the mean, std and variance of data.
    """
    
    yield [ np.mean(data), np.std(data), np.var(data) ]
    

def pool_features(data, raster_file):
    """Feature extractor for swimming pool detection.

       Args:
           data (numpy array): Pixel data vector.
           raster_file (str): Image filename.

       Yields:
           Feature vector (numpy array).
    """

    # get water signature from raster_file
    # this is hard-coded for the time being
    water_sig = np.array([399, 509, 437, 266, 219, 154, 119, 114])

    spectral_data = sa.spectral_angles(data, water_sig)

    band28_ratio = (data[1,:,:]-data[7,:,:])/(data[1,:,:]+data[7,:,:])
    
    band37_ratio = (data[2,:,:]-data[6,:,:])/(data[2,:,:]+data[6,:,:])

    yield [ band28_ratio.max(), band37_ratio.max(), np.min(spectral_data) ]
