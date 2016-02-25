from __future__ import division

import numpy as np
import timeit
import os
import sys
import geoio 


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
    if len(members) ==1:
        members.shape = (1, len(members))

    # Basic test that the data looks ok before we get going.
    assert members.shape[1] == data.shape[0], 'Data and members not of the ' \
                                              'same dimention.'

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