import geoio
import numpy as np
import geojson_tools as gt
import data_extractors as de

import warnings
warnings.filterwarnings('ignore')

def extract_polygons(train_file, min_polygon_hw = 20, max_polygon_hw = 224):
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

def get_iter_data(shapefile, batch_size=32, return_labels=False, buffer=[0,0], mask=False):

    ct, data = 0, []
    print 'Extracting image ids...'
    img_ids = gt.find_unique_values(shapefile, property_name='image_id')

    print 'Generating batch...'
    for img_id in img_ids:
        img = geoio.GeoImage(img_id + '.tif')

        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[{'image_id':img_id}],
                                                buffer=buffer,
                                                mask=mask):

            if chip is None or reduce(lambda x, y: x*y, chip.shape)==0:
                continue

            this_data = [chip, properties['feature_id']]

            if return_labels:
                try:
                    label = properties['class_name']
                    if label is None:
                        continue
                except (TypeError, KeyError):
                    continue
                this_data.append(label)
            data.append(this_data)
            ct += 1

            if ct == batch_size:
                yield zip(*data)
                ct, data = 0, []

    yield zip(*data)
