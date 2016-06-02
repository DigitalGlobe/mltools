# Contains functions for manipulating jsons and geojsons.

import geojson
import numpy as np

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


def create_balanced_geojson(shapefile, output_name, balanced = True,
                            class_names=['Swimming pool', 'No swimming pool'],
                            samples_per_class=None, train_test=None):
    '''
    Create a shapefile comprised of balanced classes for training net, and/or split
    shapefile into train and test data, each with distinct, randomly selected polygons.

    INPUT   (1) string 'shapefile': name of shapefile with original samples
            (2) string 'output_file': name of file in which to save selected polygons
            (not including file extension)
            (3) bool 'balanced': put equal amounts of each class in the output shapefile.
            Otherwise simply outputs shuffled version of original dataself.
            (4) list[string] 'class_names': name of classes of interest as listed in
            properties['class_name']. defaults to pool classes.
            (5) int or None 'samples_per_class': number of samples to select per class.
            if None, uses length of smallest class. Defaults to None
            (6) float or None 'train_test': proportion of polygons to save in test file.
            if None, only saves one file (balanced data). otherwise saves a train and
            test file. Defaults to None.

    OUTPUT  (1) geojson file with balanced classes in current directory
    '''

    with open(shapefile) as f:
        data = geojson.load(f)

    if balanced_classes:
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

    else: # dont need to ensure balanced classes
        final = data['features']

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



def filter_polygon_size(shapefile, output_file, min_polygon_hw=30, max_polygon_hw=224):
    '''
    Creates a geojson file containing only acceptable side dimensions for polygons.
    INPUT   (1) string 'shapefile': name of shapefile with original samples
            (2) string 'output_file': name of file in which to save selected polygons
            (not including file extension)
            (3) int 'min_polygon_hw': minimum acceptable side length (in pixels) for
            given polygon
            (4) int 'max_polygon_hw': maximum acceptable side length (in pixels) for
            given polygon
    OUTPUT  (1) a geojson file (output_file.geojson) containing only polygons of
            acceptable side dimensions
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

        # cycle thru polygons
        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[{'image_id': img_id}],
                                                mask=True):
            chan,h,w = np.shape(chip)
            if chip is None or min(h, w) < min_polygon_hw or max(h, w) > max_polygon_hw:
                ix += 1
                # add percent complete to stdout
                sys.stdout.write('\r%' + str(100 * ix / total) + ' ' * 20)
                sys.stdout.flush()
                continue

            ix_ok.append(ix)
            ix += 1
            # add percent complete to stdout
            sys.stdout.write('\r%{0:.2f}'.format(100 * ix / total) + ' ' * 5)
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
