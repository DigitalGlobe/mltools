import numpy as np
import data_extractors as de

import warnings
warnings.filterwarnings('ignore')

def extract_polygons(train_file, min_polygon_hw = 20, max_polygon_hw = 100):
    '''
    Create train data from shapefile, filter polygons according to acceptable side-lengths

    INPUT   (1) string 'train_file': name of shapefile containing polygon classifications
            (2) string 'target_file': name of shapefile with test data (no classifications)
            (3) int 'min_polygon_hw': minimum acceptable side length (in pixels) for polygons. Defaults to 50.
            (4) int 'max_polygon_hw': maximun accpetable side length (in pixels) for polygons. Defaults to 1000

    OUTPUT  (1) list of training rasters, zero-padded to maximum acceptable size (c, l, w)
            (2) list of ids corresponding to training rasters
            (3) list of labels corresponding to training rasters
    '''
    X_train, X_ids, X_labels = [], [], []

    # extract raw polygon rasters
    print 'Extracting raw polygons...'
    train_rasters, train_ids, train_labels = de.get_data(train_file, return_labels=True)

    # filter polygons to acceptable size.
    # can speed this up by changing de.get_data directly
    print 'Processing polygons...'
    for i in xrange(len(train_labels)):
        raster = train_rasters[i]
        c,h,w = np.shape(raster)
        if min(w, h) > min_polygon_hw and max(w, h) < max_polygon_hw:

            # zero-pad to make polygons same size
            raster = np.pad(raster, [(0,0), (0, max_polygon_hw - h), (0, max_polygon_hw - w)], 'constant', constant_values = 0)

            X_train.append(raster)
            X_ids.append(train_ids[i])
            X_labels.append(train_labels[i])
    print 'Done.'
    return X_train, X_ids, X_labels
