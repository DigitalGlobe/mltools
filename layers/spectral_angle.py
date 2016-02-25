from __future__ import division

import numpy as np
import timeit
import os
import sys
import geoio 

water_sig = np.asarray([399,509,437,266,219,154,119,114])
water_sig.shape = (1,8)

def spectral_angles_files(img_in, img_out, members):
    ''' Wrapper for spectral_angles to call files directly
    img_in = Text path to the (probably supercube) image to read data from.
    img_out = Text path for location to write an image of shape (m,x,y)
              where m is the number of spectral members to test against and
              x/y are the size of the image to test against.
    members = Python list of un-normalized spectra to test against.
    '''
    # Read in image file
    f = geoio.GeoImage(img_in) #geoio.GeoImage(img_in)
    data = f.get_data()

    # Reformat members into a numpy array
    members = np.array(members)

    # Check input data types and scale if necessary.
    if (data.dtype.kind is 'i') or (data.dtype.kind is 'u'):
        print "Well shoot, you must be working in scaled reflectance space " \
              "(0 -> 10,000) so I'll go ahead and scale the reference " \
              "spectra for you.  If this isn't what you intend, then check " \
              "the inputs/outputs closely - you'll likely be getting " \
              "bologna out."
        members = (members*10000).astype('uint16')
    elif data.dtype.kind is 'f':
        pass
    else:
        raise TypeError("The function requires numerical data.  It looks "
                        "like the input is not float or int.")

    # Get angles from numpy interface.
    angles = spectral_angles(data, members)

    # Write angles to a new file
    f.write_img_like_this(img_out,angles)
    f = None

def spectral_angles(data, members=water_sig):
    '''Pass in a numpy array of the data and a numpy array of the spectral
    members to test against.
    data = numpy array of shape (n,x,y) where n is the number of bands.
    members = numpy array of shape (m,n) where m is the number of members
              and n is the number of bands.
    '''
    # Basic test that the data looks ok before we get going.
    assert members.shape[1] == data.shape[0], 'Data and members not of the ' \
                                              'same dimention.'

    # Calculate sum of square for both data and members
    dnorm = np.linalg.norm(data,ord=2,axis=0)
    mnorm = np.linalg.norm(members,ord=2,axis=1)

    # Run angle calculations
    a = np.zeros((members.shape[0],data.shape[1],data.shape[2]))
    for m in xrange(len(mnorm)):
        num = np.sum(data*members[m,:][:,np.newaxis,np.newaxis],axis=0)
        den = dnorm*mnorm[m]
        with np.errstate(divide='ignore',invalid='ignore'):
            a[m,:,:] = num/den  #Float both from __future__ and dnorm/mnorm
            a[m,:,:] = np.arccos(np.clip(a[m,:,:],-1,1))
            a[m,:,:][den == 0] = 0

    return a

def spectral_angles_impl_tests(data, members):
    ''' Function to prototype different types of looping constructs.  The
    current "best" is copied to spectral_angles.  I'd eventually like to
    test cython as well as some of the numpy looping constructs.
    '''
    assert members.shape[1] == data.shape[0], 'Data and members not of the ' \
                                              'same dimention.'

    if 0:
        #angles = sa_impl_test_full_loop(data,members)
        print "Pure python loop:"
        print timeit.repeat("sa_impl_test_full_loop(data,members)",
                            "from __main__ import calc_along_z_loop,d",
                            repeat=num,
                            number=1)
        print ''



    if 1:
        #angles = sa_impl_test_numpy_loopbands(data,members)
        print "Numpy loop bands:"
        print timeit.repeat("sa_impl_test_numpy_loopbands(data,members)",
                            "from __main__ import calc_along_z_sum,d",
                            repeat=num,
                            number=1)
        print ''

    if 0:
        #angles = sa_impl_test_numpy_noloops(data,members)
        print "Numpy no loops:"
        print timeit.repeat("sa_impl_test_numpy_noloops(data,members)",
                            "from __main__ import calc_along_z_sum,d",
                            repeat=num,
                            number=1)
        print ''

    if 0:
        angles = sa_impl_test_np_iterator(data,members)

    if 0:
        angles = sa_impl_test_cython(data,members)

def sa_impl_test_full_loop(data,members):
    #### Slow version ####
    # 9 min 31 seconds per loop on 16x1000x1000 data with 5 member spectra
    # For each member calculate angle
    a = np.zeros((members.shape[1],data.shape[1],data.shape[2]))
    for x in xrange(data.shape[1]):
        for y in xrange(data.shape[2]):
            for m in xrange(members.shape[1]):
                t = data[:,x,y]
                r = members[:,m]
                a[m,x,y] = np.arccos(
                            (np.sum(t*r))/
                            (np.sqrt(np.sum(t**2))*np.sqrt(np.sum(r**2))))

    return a

def sa_impl_test_numpy_loopbands(data,members):
    #### Numpy version - Loop the bands ####
    # 5.46 seconds per loop on 16x1000x1000 data with 5 member spectra
    # Calculate sum of square for both data and members
    dnorm = np.linalg.norm(data,ord=2,axis=0)
    mnorm = np.linalg.norm(members,ord=2,axis=1)

    # Run angle calculations
    a = np.zeros((members.shape[0],data.shape[1],data.shape[2]))
    for m in xrange(len(mnorm)):
        num = np.sum(data*members[m,:][:,np.newaxis,np.newaxis],axis=0)
        den = dnorm*mnorm[m]
        a[m,:,:] = np.arccos(np.clip(num/den,-1,1))

    return a

def sa_impl_test_numpy_noloops(data,members):
    #### Numpy version 2 ####
    # 5.54 seconds per loop on 16x1000x1000 data with 5 member spectra
    # 5.35 seconds per loop if you don't have to swap axes
    #########################

    # Calculate sum of square for both data and members
    dnorm = np.linalg.norm(data,ord=2,axis=0)
    mnorm = np.linalg.norm(members,ord=2,axis=1)

    num = np.sum(data[np.newaxis,:,:,:]*
                 members[:,:,np.newaxis,np.newaxis],axis=1)
    den = dnorm[np.newaxis,:,:]*mnorm[:,np.newaxis,np.newaxis]
    a = np.arccos(np.clip(num/den,-1,1))

    return a

def sa_impl_test_np_iterator(data,members):
    #### Numpy iterator ####
    # ??? seconds per loop on 16x1000x1000 data with 5 member spectra
    a = np.zeros((members.shape[1],data.shape[1],data.shape[2]))
    for x in np.nditer(data.shape[1]):
        for y in np.nditer(data.shape[2]):
            for m in np.nditer(members.shape[1]):
                print 'something'

    return a

def sa_impl_test_cython(data,members):
    #### Cython loops ####
    # ??? seconds per loop on 16x1000x1000 data with 5 member spectra
    pass
    a = 1

    return a

def test_numpy_interface(run_impl_tests=False):
    '''Quick test of the spectral_angles numpy interface.'''
    size_set = 2
    nbands = 16
    memb_size = 5

    if size_set == 3:
        x = 12000
        y = 16000

    if size_set == 2:
        x = 1000
        y = 2000

    if size_set == 1:
        x = 100
        y = 100

    # Make fake data array
    # data = np.random.randn(nbands,x,y)
    data = np.random.randint(low=1,high=2048,size=(nbands,x,y))

    # Make fake spectral members
    # members = np.random.randn(memb_size,nbands)
    members = np.random.randint(low=1,high=2048,size=(memb_size,nbands))

    # Run spectral_angles from numpy arrays directly
    if run_impl_tests:
        angles = spectral_angles_impl_tests(data, members)
    else:
        angles = spectral_angles(data, members)

    for x in [0,10,40,54,78,90,92]:
        # print('Running test pixels...')
        t = data[:,x,x]
        r = members[0,:]
        tmp=np.arccos(np.clip(
            (np.sum(t*r))/(np.sqrt(np.sum(t**2))*np.sqrt(np.sum(r**2)))
                            ,-1,1))
        # print tmp
        assert angles[0,x,x]==tmp
        tn = np.linalg.norm(t,2)
        rn = np.linalg.norm(r,2)
        tmp=np.arccos(np.clip(t.dot(r)/np.dot(tn,rn),-1,1))
        # print tmp
        assert angles[0,x,x]==tmp
    print("Output looks good... I think. =)")

def  test_files_interface():

    nbands = 8
    memb_size = 5

    img_in = '/mnt/panasas/nwl/code/Campyng/campyng/data/imagefiles' \
             '/smalldgdelivery/053792616010_01/053792616010_01_P001_MUL/' \
             '14JUN20181517-M2AS-053792616010_01_P001.TIF'
    img_out = '/mnt/panasas/nwl/code/Campyng/campyng/tests/testsoutput/' \
             'spectral_angles_testout.TIF'
    member_list = []
    for i in xrange(memb_size):
        member_list.append(np.random.randn(nbands))

    angles = spectral_angles_files(img_in,img_out,member_list)

if __name__== "__main__":

    # Run spectral_angles from made up numpy arrays
    if 0:
        test_numpy_interface()

    # Run spectral_angles from image files
    if 1:
        test_numpy_interface()

    # Run spectral_angles from image files
    if 0:
        test_numpy_interface(run_impl_tests=True)
