import geoio
import geojson
import random
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


def get_iter_data(shapefile, batch_size=32, return_labels=True, buffer=[0,0], mask=True):
    '''
    Generates batches of training data from shapefile for when it will not fit in memory.

    INPUT   (1) string 'shapefile': name of shapefile to extract polygons from
            (2) int 'batch_size': number of chips to generate per iteration. equal to batch-size of net, defaults to 32
            (3) bool 'return_labels': return class label with chips. defaults to True
            (4) list[int] 'buffer': two-dim buffer in pixels. defaults to [0,0].
            (5) bool 'mask': if True returns a masked array. defaults to True

    OUTPUT  (1) chips: one batch of masked (if True) chips
            (2) corresponding feature_id for chips
            (3) corresponding chip labels (if True)
    '''

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

def create_balanced_geojson(shapefile, output_name, class_names=['Swimming pool', 'No swimming pool'], samples_per_class = None):
    '''
    Create a shapefile comprised of balanced classes for training net

    INPUT   (1) string 'shapefile': name of shapefile with original samples
            (2) string 'output_file': name of file in which to save selected polygons (not including file extension)
            (3) list[string] 'class_names': name of classes of interest as listed in properties['class_name']. defaults to pool classes.
            (4) int or None 'samples_per_class': number of samples to select per class. if None, uses length of smallest class. Defaults to None

    OUTPUT  (1) geojson file with balanced classes in current directory
    '''
    sorted_classes = [] # put different classes in separate lists

    with open(shapefile) as f:
        data=geojson.load(f)

    # separate different classes based on class_names
    for i in class_names:
        this_data = []

        for feat in data['features']:
            if feat['properties']['class_name'] == i:
                this_data.append(feat)

        sorted_classes.append(this_data)

    # randomly select given number of samples per class
    if samples_per_class:
        samples = [random.sample(i, samples_per_class) for i in sorted_classes]
        final = [s for sample in samples for s in sample]

    else:
        # determine smallest class-size
        small_class_ix = np.argmin([len(clss) for clss in sorted_classes])
        class_sizes = len(sorted_classes[small_class_ix])
        final = sorted_classes[small_class_ix]

        # randomly sample from larger classes to balance class sizes
        for i in xrange(len(class_names)):
            if i == small_class_ix:
                continue
            else:
                final += random.sample(sorted_classes[i], class_sizes)

    # shuffle classes for input to net
    np.random.shuffle(final)
    balanced_json = {data.keys()[0]: data.values()[0], data.keys()[1]: final}

    with open(output_name + '.geojson', 'wb') as f:
        geojson.dump(balanced_json, f)
