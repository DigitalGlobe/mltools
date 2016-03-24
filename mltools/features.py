# Contains feature functions which can be used
# to construct feature vectors.

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
    return (data[band1-1,:,:] - data[band2-1,:,:])/(data[band1-1,:,:] + data[band2-1,:,:])


