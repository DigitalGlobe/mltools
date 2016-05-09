# Feature functions for feature vector construction.

from __future__ import division
import numpy as np


def spectral_angles(data, members):
    '''Compute spectral_angles between data and the spectral profiles in members.
    
       Args: 
           data (numpy array): Array of shape (n,x,y) where n is band number.
           members (numpy array): Array of shape (m,n) where m is member number
                                  and n is band number.

       Returns: Spectral angle vector (numpy array)                                   
    '''

    # if members is one-dimensional, convert to horizontal vector 
    if len(members.shape) ==1:
        members.shape = (1, len(members))

    # Basic test that the data looks ok before we get going.
    assert members.shape[1] == data.shape[0], 'Dimension conflict!'

    # Calculate sum of square for both data and members
    dnorm = np.linalg.norm(data, ord=2, axis=0)
    mnorm = np.linalg.norm(members, ord=2, axis=1)

    # Run angle calculations
    a = np.zeros((members.shape[0], data.shape[1], data.shape[2]))
    for m in xrange(len(mnorm)):
        num = np.sum(data * members[m, :][:, np.newaxis, np.newaxis], axis=0)
        den = dnorm * mnorm[m]
        with np.errstate(divide='ignore', invalid='ignore'):
            a[m, :, :] = num/den  #Float both from __future__ and dnorm/mnorm
            a[m, :, :] = np.arccos(np.clip(a[m,:,:],-1,1))
            a[m, :, :][den == 0] = 0

    return a


def band_ratios(data, band1, band2):
    '''Returns ratio = (band1 - band2)/(band1 + band2) for every pixel in data.

       Args: 
           data (numpy array): Array of shape (n,x,y) where n is band number.
           band1 (int): band1 index (from 1 to n)
           band2 (int): band2 index (from 1 to n) 
      
    '''
    data = np.array(data, dtype=float)
    return (data[band1-1,:,:] - data[band2-1,:,:])/(data[band1-1,:,:] + data[band2-1,:,:])


def pool_basic(data):
    '''Feature vector for swimming pool detection.
       Args:
           data (numpy array): Pixel data vector.          
       Returns:
           Feature numpy vector.
    '''

    # pool signatures from acomped imagery of adelaide, australia
    pool_sig = np.array([1179, 2295, 2179, 759, 628, 186, 270, 110])
    covered_pool_sig = np.array([1584, 1808, 1150, 1104, 1035, 995, 1659, 1741])

    pool_data = spectral_angles(data, pool_sig)
    covered_pool_data = spectral_angles(data, covered_pool_sig)
    band26_ratio = band_ratios(data, 2, 6)
    band36_ratio = band_ratios(data, 3, 6)

    return [np.max(band26_ratio), np.max(band36_ratio), np.min(pool_data), np.min(covered_pool_data)]
