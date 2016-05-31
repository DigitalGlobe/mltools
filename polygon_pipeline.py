import geoio
import geojson
import random
import numpy as np
import sys
# import geojson_tools as gt
from skimage.transform import resize
from keras.utils import np_utils
from mltools import geojson_tools as gt
# from mltools import data_extractors as de

import warnings
warnings.filterwarnings('ignore')


def get_iter_data(shapefile, batch_size=32, nb_classes=2, min_chip_hw=40,
                  max_chip_hw=224, return_labels=True, buffer=[0, 0], mask=True, fc=False,
                  resize_dim=None, normalize=False):
    '''
    Generates batches of training data from shapefile for when it will not fit in memory.
    INPUT   (1) string 'shapefile': name of shapefile to extract polygons from
            (2) int 'batch_size': number of chips to generate per iteration. equal to
            batch-size of net, defaults to 32
            (3) int 'nb_classes': number of classes in which to categorize itmes
            (4) int 'min_chip_hw': minimum size acceptable (in pixels) for a polygon.
            defaults to 100
            (5) int 'max_chip_hw': maximum size acceptable (in pixels) for a polygon.
            note that this will be the size of the height and width of input images to the
            net (default = 224)
            (6) bool 'return_labels': return class label with chips. defaults to True
            (7) list[int] 'buffer': two-dim buffer in pixels. defaults to [0,0].
            (8) bool 'mask': if True returns a masked array. defaults to True
            (9) bool 'fc': return appropriately shaped target vector for FCNN
            (10) tuple(int) 'resize_dim': size to downsample chips to (channels, height,
            width). Note that resizing takes place after padding the original polygon.
            Defaults to None (do not resize).
            (11) bool 'normalize': divide all chips by max pixel intensity (normalize net input)
    OUTPUT  (1) chips: one batch of masked (if True) chips
            (2) corresponding feature_id for chips
            (3) corresponding chip labels (if True)
    '''

    ct, inputs, labels = 0, [], []
    print 'Extracting image ids...'
    img_ids = gt.find_unique_values(shapefile, property_name='image_id')

    for img_id in img_ids:
        img = geoio.GeoImage(img_id + '.tif')

        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[{'image_id': img_id}],
                                                buffer=buffer,
                                                mask=mask):

            # check for adequate chip size
            chan, h, w = np.shape(chip)
            if chip is None or min(h, w) < min_chip_hw or max(
                    h, w) > max_chip_hw:
                continue

            # zero-pad chip to standard net input size
            chip = chip.filled(0)[:3].astype(float)  # replace masked entries with zeros
            chip_patch = np.pad(chip, [(0, 0), (1 - ((max_chip_hw - h)/2)), ((max_chip_hw - h)/2), (1 - ((max_chip_hw - w)/2)), ((max_chip_hw - w)/2)], 'constant', constant_values=0)

            # resize image
            if resize_dim:
                if resize_dim != chip_patch.shape:
                    chip_patch = resize(chip_patch, resize_dim)

            if normalize:
                chip_patch /= np.max(chip_patch)

            if return_labels:
                try:
                    label = properties['class_name']
                    if label is None:
                        continue
                    labels.append(label)
                except (TypeError, KeyError):
                    continue

            # do not include image_id for fitting net
            inputs.append(chip_patch)
            ct += 1
            if ct == batch_size:
                l = [1 if lab == 'Swimming pool' else 0 for lab in labels]
                labels = np_utils.to_categorical(l, nb_classes)
                # reshape label vector to match output of FCNN
                if not fc:
                    yield (np.array([i[:3] for i in inputs]), labels)
                else:
                    yield (np.array([i[:3] for i in inputs]), labels.reshape(batch_size, nb_classes, 1))
                ct, inputs, labels = 0, [], []

    # return any remaining inputs
    if len(inputs) != 0:
        l = [1 if lab == 'Swimming pool' else 0 for lab in labels]
        labels = np_utils.to_categorical(l, 2)
        yield (np.array([i[:3] for i  in inputs]), np.array(labels))


def filter_polygon_size(shapefile, output_file, min_polygon_hw=30, max_polygon_hw=224):
    '''
    Creates a geojson file containing only acceptable side dimensions for polygons.
    INPUT   (1) string 'shapefile': name of shapefile with original samples
            (2) string 'output_file': name of file in which to save selected polygons
            (not including file extension)
            (3) int 'min_polygon_hw': minimum acceptable side length (in pixels) for given polygon
            (4) int 'max_polygon_hw': maximum acceptable side length (in pixels) for given polygon
    OUTPUT  (1) a geojson file (output_file.geojson) containing only polygons of acceptable side dimensions
    '''
    # load polygons
    with open(shapefile) as f:
        data = geojson.load(f)
    total = float(len(data['features']))

    # find indicies of acceptable polygons
    ix_ok, ix = [], 0
    print 'Extracting image ids...'
    img_ids = gt.find_unique_values(shapefile, property_name='image_id')

    print 'Filtering polygons...'
    for img_id in img_ids:
        print '... for image {}'.format(img_id)
        img = geoio.GeoImage(img_id + '.tif')

        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[{'image_id': img_id}],
                                                mask=True):
            chan,h,w = np.shape(chip)
            if chip is None or min(h, w) < min_polygon_hw or max(h, w) > max_polygon_hw:
                ix += 1
                sys.stdout.write('\r%' + str(100 * ix / total) + ' ' * 20)
                sys.stdout.flush()
                continue

            ix_ok.append(ix)
            ix += 1
            sys.stdout.write('\r%' + str(100 * ix / total) + ' ' * 20)
            sys.stdout.flush()

    print 'Saving...'
    ok_polygons = [data['features'][i] for i in ix_ok]
    np.random.shuffle(ok_polygons)
    filtrate = {data.keys()[0]: data.values()[0],
                data.keys()[1]: ok_polygons}

    # save new geojson
    with open('{}.geojson'.format(output_file), 'wb') as f:
        geojson.dump(filtrate, f)

    print 'Saved {} polygons to {}.geojson'.format(len(ok_polygons), output_file)


def create_balanced_geojson(shapefile, output_name,
                            class_names=['Swimming pool', 'No swimming pool'],
                            samples_per_class=None, train_test=None):
    '''
    Create a shapefile comprised of balanced classes for training net. Option to save a
    train and test file- each with distinct, randomly selected polygons.

    INPUT   (1) string 'shapefile': name of shapefile with original samples
            (2) string 'output_file': name of file in which to save selected polygons
            (not including file extension)
            (3) list[string] 'class_names': name of classes of interest as listed in
            properties['class_name']. defaults to pool classes.
            (4) int or None 'samples_per_class': number of samples to select per class.
            if None, uses length of smallest class. Defaults to None
            (5) float or None 'train_test': proportion of polygons to save in test file.
            if None, only saves one file (balanced data). otherwise saves a train and
            test file. Defaults to None.

    OUTPUT  (1) geojson file with balanced classes in current directory
    '''

    with open(shapefile) as f:
        data = geojson.load(f)

    # sort classes into separate lists
    sorted_classes = []

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

    # split feature lists into train and test
    if train_test:
        test_out = 'test_{}'.format(output_name + '.geojson')
        train_out = 'train_{}'.format(output_name + '.geojson')
        test_size = int(train_test * len(final))
        test = {
            data.keys()[0]: data.values()[0],
            data.keys()[1]: final[:test_size]}
        train = {
            data.keys()[0]: data.values()[0],
            data.keys()[1]: final[test_size:]}

        # save train and test geojsons
        with open(test_out, 'wb') as f1:
            geojson.dump(test, f1)
        print 'Test polygons saved as {}'.format(test_out)

        with open(train_out, 'wb') as f2:
            geojson.dump(train, f2)
        print 'Train polygons saved as {}'.format(train_out)

    else:  # only save one file with balanced classes
        balanced_json = {
            data.keys()[0]: data.values()[0],
            data.keys()[1]: final}
        with open(output_name + '.geojson', 'wb') as f:
            geojson.dump(balanced_json, f)
        print '{} polygons saved as {}.geojson'.format(len(final), output_name)


## GRAVEYARD ##

# in get_iter_data:
    # # return any remaining inputs
    # if len(inputs) != 0:
    #     l = [1 if lab == 'Swimming pool' else 0 for lab in labels]
    #     labels = np_utils.to_categorical(l, 2)
    #     yield (np.array([i[:3] for i  in inputs]), np.array(labels))


# def extract_polygons(train_file, min_polygon_hw = 20, max_polygon_hw = 224):
#     '''
#     Create train data from shapefile, filter polygons according to acceptable side-lengths
#
#     INPUT   (1) string 'train_file': name of shapefile containing polygon classifications
#             (2) string 'target_file': name of shapefile with test data (no classifications)
#             (3) int 'min_polygon_hw': minimum acceptable side length (in pixels) for polygons. Defaults to 50.
#             (4) int 'max_polygon_hw': maximun accpetable side length (in pixels) for polygons. Defaults to 1000
#
#     OUTPUT  (1) list of training rasters, zero-padded to maximum acceptable size (c, l, w)
#             (2) list of ids corresponding to training rasters
#             (3) list of labels corresponding to training rasters
#     '''
#     X_train, X_ids, X_labels = [], [], []
#
#     # extract raw polygon rasters
#     print 'Extracting raw polygons...'
#     train_rasters, train_ids, train_labels = de.get_data(train_file, return_labels=True)
#
#     # filter polygons to acceptable size.
#     print 'Processing polygons...'
#     for i in xrange(len(train_labels)):
#         raster = train_rasters[i]
#         c,h,w = np.shape(raster)
#         if min(w, h) > min_polygon_hw and max(w, h) < max_polygon_hw:
#
#             # zero-pad to make polygons same size
#             raster = np.pad(raster, [(0,0), (0, max_polygon_hw - h), (0, max_polygon_hw - w)], 'constant', constant_values = 0)
#
#             X_train.append(raster)
#             X_ids.append(train_ids[i])
#             X_labels.append(train_labels[i])
#     print 'Done.'
#     return X_train, X_ids, X_labels
