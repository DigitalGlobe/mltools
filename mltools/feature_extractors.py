"""
What: Contains various feature extraction functions.
Authors: Carsten Tusk, Kostas Stamatiou, Nathan Longbotham
Contact: kostas.stamatiou@digitalglobe.com
"""

from __future__ import division
import numpy as np


def spectral_angles(data, members):
    """Pass in a numpy array of the data and a numpy array of the spectral
       members to test against.
    
       Args: 
           data (numpy array): Array of shape (n,x,y) where n is band number.
           members (numpy array): Array of shape (m,n) where m is member number
                                  and n is the number of bands.

       Returns: Spectral angle vector (numpy array)
                                      
    """

    # if members is one-dimensional, convert to horizontal vector 
    if len(members.shape) ==1:
        members.shape = (1, len(members))

    # Basic test that the data looks ok before we get going.
    assert members.shape[1] == data.shape[0], 'Dimension conflict!'

    # Calculate sum of square for both data and members
    dnorm = np.linalg.norm(data,ord=2,axis=0)
    mnorm = np.linalg.norm(members,ord=2,axis=1)

    # Run angle calculations
    a = np.zeros((members.shape[0], data.shape[1], data.shape[2]))
    for m in xrange(len(mnorm)):
        num = np.sum(data*members[m,:][:,np.newaxis,np.newaxis],axis=0)
        den = dnorm*mnorm[m]
        with np.errstate(divide='ignore', invalid='ignore'):
            a[m,:,:] = num/den  #Float both from __future__ and dnorm/mnorm
            a[m,:,:] = np.arccos(np.clip(a[m,:,:],-1,1))
            a[m,:,:][den == 0] = 0

    return a


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
    # water_sig = np.array([399, 509, 437, 266, 219, 154, 119, 114])
    water_sig = np.array([393, 465, 473, 271, 173, 76, 47, 25])

    covered_pool_sig = np.array([404, 463, 430, 332, 276, 268, 587, 394])

    spectral_data = spectral_angles(data, water_sig)
    spectral_data_2 = spectral_angles(data, covered_pool_sig)

    #band28_ratio = (data[1,:,:]-data[7,:,:])/(data[1,:,:]+data[7,:,:])
    
    #band37_ratio = (data[2,:,:]-data[6,:,:])/(data[2,:,:]+data[6,:,:])

    #return [ band28_ratio.max(), band37_ratio.max(), np.min(spectral_data), np.min(spectral_data_2) ]
    return [np.min(spectral_data), np.min(spectral_data_2) ]
