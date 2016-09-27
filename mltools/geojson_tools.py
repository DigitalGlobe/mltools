# Contains functions for manipulating jsons and geojsons.

import geojson
import numpy as np
import geoio
import sys
import random
import subprocess
import os

from shapely.wkb import loads


def join(input_files, output_file):
    """Join geojsons into one. The spatial reference system of the
       output file is the same as the one of the last file in the list.

       Args:
           input_files (list): List of file name strings.
           output_file (str): Output file name.
    """

    # get feature collections
    final_features = []
    for file in input_files:
        with open(file) as f:
            feat_collection = geojson.load(f)
            final_features += feat_collection['features']

    feat_collection['features'] = final_features

    # write to output file
    with open(output_file, 'w') as f:
        geojson.dump(feat_collection, f)


def split(input_file, file_1, file_2, no_in_first_file):
    """Split a geojson in two separate files.

       Args:
           input_file (str): Input filename.
           file_1 (str): Output file name 1.
           file_2 (str): Output file name 2.
           no_features (int): Number of features in input_file to go to file_1.
           output_file (str): Output file name.
    """

    # get feature collection
    with open(input_file) as f:
        feat_collection = geojson.load(f)

    features = feat_collection['features']
    feat_collection_1 = geojson.FeatureCollection(features[0:no_in_first_file])
    feat_collection_2 = geojson.FeatureCollection(features[no_in_first_file:])

    with open(file_1, 'w') as f:
        geojson.dump(feat_collection_1, f)

    with open(file_2, 'w') as f:
        geojson.dump(feat_collection_2, f)


def get_from(input_file, property_names):
    """Reads a geojson and returns a list of value tuples, each value
       corresponding to a property in property_names.

       Args:
           input_file (str): File name.
           property_names: List of strings; each string is a property name.

       Returns:
           List of value tuples.
    """

    # get feature collections
    with open(input_file) as f:
        feature_collection = geojson.load(f)

    features = feature_collection['features']
    values = [tuple([feat['properties'].get(x)
                     for x in property_names]) for feat in features]

    return values


def write_to(data, property_names, output_file):
    '''Write list of tuples to geojson.
       First entry of each tuple should be geometry in hex coordinates
       and the rest properties.

       Args:
           data: List of tuples.
           property_names: List of strings. Should be same length as the
                           number of properties.
           output_file (str): Output file name.

    '''

    geojson_features = []
    for entry in data:
        coords_in_hex, properties = entry[0], entry[1:]
        geometry = loads(coords_in_hex, hex=True)
        property_dict = dict(zip(property_names, properties))
        if geometry.geom_type == 'Polygon':
            coords = [list(geometry.exterior.coords)]   # brackets required
            geojson_feature = geojson.Feature(geometry=geojson.Polygon(coords),
                                              properties=property_dict)
        elif geometry.geom_type == 'Point':
            coords = list(geometry.coords)[0]
            geojson_feature = geojson.Feature(geometry=geojson.Point(coords),
                                              properties=property_dict)
        geojson_features.append(geojson_feature)

    feature_collection = geojson.FeatureCollection(geojson_features)

    with open(output_file, 'wb') as f:
        geojson.dump(feature_collection, f)


def write_properties_to(data, property_names, input_file,
                        output_file, filter=None):
    """Writes property data to polygon_file for all
       geometries indicated in the filter, and creates output file.
       The length of data must be equal to the number of geometries in
       the filter. Existing property values are overwritten.

       Args:
           data (list): List of tuples. Each entry is a tuple of dimension equal
                        to property_names.
           property_names (list): Property names.
           input_file (str): Input file name.
           output_file (str): Output file name.
           filter (dict): Filter format is {'property_name':[value1,value2,...]}.
                          What this achieves is to write the first entry of data
                          to the properties of the feature with
                          'property_name'=value1, and so on. This makes sense only
                          if these values are unique. If Filter=None, then
                          data is written to all geometries in the input file.
    """

    with open(input_file) as f:
        feature_collection = geojson.load(f)

    features = feature_collection['features']

    if filter is None:
        for i, feature in enumerate(features):
            for j, property_value in enumerate(data[i]):
                feature['properties'][property_names[j]] = property_value
    else:
        filter_name = filter.keys()[0]
        filter_values = np.array(filter.values()[0])
        for feature in features:
            compare_value = feature['properties'][filter_name]
            ind = np.where(filter_values == compare_value)[0]
            if len(ind) > 0:
                for j, property_value in enumerate(data[ind]):
                    feature['properties'][property_names[j]] = property_value

    feature_collection['features'] = features

    with open(output_file, 'w') as f:
        geojson.dump(feature_collection, f)


def find_unique_values(input_file, property_name):
    """Find unique values of a given property in a geojson file.

       Args:
           input_file (str): File name.
           property_name (str): Property name.

       Returns:
           List of distinct values of property.
           If property does not exist, it returns None.
    """
    with open(input_file) as f:
        feature_collection = geojson.load(f)

    features = feature_collection['features']
    values = np.array([feat['properties'].get(property_name)
                       for feat in features])
    return np.unique(values)


def create_balanced_geojson(shapefile, output_file, balanced = True,
                            class_names=['Swimming pool', 'No swimming pool'],
                            samples_per_class=None, train_test=None):
    '''
    Create a shapefile comprised of balanced classes for training net, and/or split
    shapefile into train and test data, each with distinct, randomly selected polygons.

    INPUT   (1) string 'shapefile': name of shapefile with original samples
            (2) string 'output_file': name of file in which to save selected polygons.
            This should end in '.geojson'
            (3) bool 'balanced': put equal amounts of each class in the output shapefile.
            Otherwise simply outputs shuffled version of original dataself.
            (4) list[string] 'class_names': name of classes of interest as listed in
            properties['class_name']. defaults to pool classes.
            (5) int or None 'samples_per_class': number of samples to select per class.
            if None, uses length of smallest class. Defaults to None
            (6) float or None 'train_test': proportion of polygons to save in test file.
            if None, only saves one file (balanced data). otherwise saves a train and
            test file. Defaults to None.

    OUTPUT  (1) train geojson file with balanced classes (if True) in current directory.
            (2) test geojson file if train_test is specified
    '''

    with open(shapefile) as f:
        data = geojson.load(f)

    if balanced:
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

    else: # don't need to ensure balanced classes
        final = data['features']

    # shuffle classes for input to net
    np.random.shuffle(final)

    # split feature lists into train and test
    if train_test:
        test_out = 'test_{}'.format(output_file)
        train_out = 'train_{}'.format(output_file)
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
        with open(output_file, 'wb') as f:
            geojson.dump(balanced_json, f)
        print '{} polygons saved as {}'.format(len(final), output_file)



def filter_polygon_size(shapefile, output_file, min_polygon_hw=0, max_polygon_hw=125,
                        shuffle=False, make_omitted_files=False):
    '''
    Creates a geojson file containing only acceptable side dimensions for polygons.
    INPUT   (1) string 'shapefile': name of shapefile with original samples
            (2) string 'output_file': name of file in which to save selected polygons.
            (3) int 'min_polygon_hw': minimum acceptable side length (in pixels) for
                given polygon
            (4) int 'max_polygon_hw': maximum acceptable side length (in pixels) for
                given polygon
            (5) bool 'shuffle': shuffle polygons before saving to output file. Defaults to
                False
            (6) bool 'make_omitted_files': create a file with omitted polygons. Two files
                will be created, one with polygons that are too small and one with large
                polygons. Defaults to False.
    OUTPUT  (1) a geojson file (output_file.geojson) containing only polygons of
                acceptable side dimensions
    '''
    # load polygons
    with open(shapefile) as f:
        data = geojson.load(f)
    total_features = float(len(data['features']))

    # format output file name
    if output_file[-8:] != '.geojson':
        output_file = output_file + '.geojson'

    # find indicies of acceptable polygons
    ix_ok, small_ix, large_ix = [], [], []
    print 'Extracting image ids...'
    img_ids = find_unique_values(shapefile, property_name='image_id')

    print 'Filtering polygons...'
    for img_id in img_ids:
        ix = 0
        print '... for image {}'.format(img_id)
        img = geoio.GeoImage(img_id + '.tif')

        # create vrt if img has multiple bands (more efficient)
        if img.shape[0] > 1:
            vrt_cmd = 'gdalbuildvrt tmp.vrt -b 1 {}.tif'.format(img_id)
            subprocess.call(vrt_cmd, shell=True) #saves temporary vrt file to filter on
            img = geoio.GeoImage('tmp.vrt')

        # cycle thru polygons
        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[{'image_id': img_id}],
                                                mask=True):
            if chip is None:
                ix += 1
                # add percent complete to stdout
                sys.stdout.write('\r%{0:.2f}'.format(100 * ix / total_features) + ' ' * 20)
                sys.stdout.flush()
                continue

            chan,h,w = np.shape(chip)

            # Identify small chips
            if min(h, w) < min_polygon_hw:
                small_ix.append(ix)
                ix += 1
                sys.stdout.write('\r%{0:.2f}'.format(100 * ix / total_features) + ' ' * 20)
                sys.stdout.flush()
                continue

            # Identify large chips
            elif max(h, w) > max_polygon_hw:
                large_ix.append(ix)
                ix += 1
                sys.stdout.write('\r%{0:.2f}'.format(100 * ix / total_features) + ' ' * 20)
                sys.stdout.flush()
                continue

            # Identify valid chips
            ix_ok.append(ix)
            ix += 1
            # add percent complete to stdout
            sys.stdout.write('\r%{0:.2f}'.format(100 * ix / total_features) + ' ' * 20)
            sys.stdout.flush()

        # remove vrt file
        try:
            os.remove('tmp.vrt')
        except:
            pass

    # save new geojson
    print 'Saving...'
    ok_polygons = [data['features'][i] for i in ix_ok]
    small_polygons = [data['features'][i] for i in small_ix]
    large_polygons = [data['features'][i] for i in large_ix]
    print str(len(small_polygons)) + ' small polygons removed'
    print str(len(large_polygons)) + ' large polygons removed'

    if shuffle:
        np.random.shuffle(ok_polygons)

    filtrate = {data.keys()[i]: data.values()[i] for i in xrange(len(data.keys()) - 1)}
    filtrate['features'] = ok_polygons

    with open(output_file, 'wb') as f:
        geojson.dump(filtrate, f)

    if make_omitted_files:
        # make file with small polygons
        small = {data.keys()[i]: data.values()[i] for i in xrange(len(data.keys()) - 1)}
        small['features'] = small_polygons
        with open('small_' + output_file, 'w') as f:
            geojson.dump(small, f)

        # make file with large polygons
        large = {data.keys()[i]: data.values()[i] for i in xrange(len(data.keys()) - 1)}
        large['features'] = large_polygons
        with open('large_' + output_file, 'w') as f:
            geojson.dump(large, f)

    print 'Saved {} polygons to {}'.format(str(len(ok_polygons)), output_file)
