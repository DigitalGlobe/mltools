# Extract pixels and metadata using shapefiles and georeferenced imagery.
# The purpose of this module is to generate train, test and target data
# for machine learning algorithms.

import geoio
import geojson
import geojson_tools as gt
import numpy as np
import sys
from itertools import cycle
import osgeo.gdal as gdal
from osgeo.gdalconst import *
from functools import reduce


def get_data(shapefile, return_labels=False, return_id=False, buffer=[0, 0], mask=False,
             num_chips=None):
    """Return pixel intensity array for each geometry in shapefile.
       The image reference for each geometry is found in the image_id
       property of the shapefile.
       If shapefile contains points, then buffer must have non-zero entries.
       The function also can also return a list of geometry ids; this is useful in
       case some of the shapefile entries do not produce a valid intensity
       array and/or class name.

       Args:
           shapefile (str): Name of shapefile in mltools geojson format.
           return_labels (bool): If True, then a label vector is returned.
           return_id (bool): if True, then the geometry id is returned.
           buffer (list): 2-dim buffer in PIXELS. The size of the box in each
                          dimension is TWICE the buffer size.
           mask (bool): Return a masked array.
           num_chips (int): Maximum number of arrays to return.

       Returns:
           chips (list): List of pixel intensity numpy arrays.
           ids (list): List of corresponding geometry ids.
           labels (list): List of class names, if return_labels=True
    """

    data, ct = [], 0

    # go through point_file and unique image_id's
    image_ids = gt.find_unique_values(shapefile, property_name='image_id')

    # go through the shapefile for each image --- this is how geoio works
    for image_id in image_ids:

        # add tif extension
        img = geoio.GeoImage(image_id + '.tif')

        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[
                                                    {'image_id': image_id}],
                                                buffer=buffer,
                                                mask=mask):

            if chip is None or reduce(lambda x, y: x * y, chip.shape) == 0:
                continue

            # every geometry must have id
            if return_id:
                this_data = [chip, properties['feature_id']]
            else:
                this_data = [chip]

            if return_labels:
                try:
                    label = properties['class_name']
                    if label is None:
                        continue
                except (TypeError, KeyError):
                    continue
                this_data.append(label)

            data.append(this_data)

            # return if max num chips is reached
            if num_chips:
                ct += 1
                if ct == num_chips:
                    return zip(*data)

    return zip(*data)


def random_window(image, chip_size, no_chips=10000):
    """Implement a random chipper on a georeferenced image.

       Args:
           image (str): Image filename.
           chip_size (list): Array of chip dimensions.
           no_chips (int): Number of chips.

       Returns:
           List of chip rasters.
    """
    img = geoio.GeoImage(image)

    chips = []
    for i, chip in enumerate(img.iter_window_random(
            win_size=chip_size, no_chips=no_chips)):
        chips.append(chip)
        if i == no_chips - 1:
            break

    return chips


def apply_mask(input_file, mask_file, output_file):
    """Apply binary mask on image. Input image and mask must have the same
       (x,y) dimension and the same projection.

       Args:
           input_file (str): Input file name.
           mask_file (str): Mask file name.
           output_file (str): Masked image file name.
    """

    source_ds = gdal.Open(input_file, GA_ReadOnly)
    nbands = source_ds.RasterCount
    mask_ds = gdal.Open(mask_file, GA_ReadOnly)

    xsize, ysize = source_ds.RasterXSize, source_ds.RasterYSize
    xmasksize, ymasksize = mask_ds.RasterXSize, mask_ds.RasterYSize

    print 'Generating mask'

    # Create target DS
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_file, xsize, ysize, nbands, GDT_Byte)
    dst_ds.SetGeoTransform(source_ds.GetGeoTransform())
    dst_ds.SetProjection(source_ds.GetProjection())

    # Apply mask --- this is line by line at the moment, not so efficient
    for i in range(ysize):
        # read line from input image
        line = source_ds.ReadAsArray(xoff=0, yoff=i, xsize=xsize, ysize=1)
        # read line from mask
        mask_line = mask_ds.ReadAsArray(xoff=0, yoff=i, xsize=xsize, ysize=1)
        # apply mask
        masked_line = line * (mask_line > 0)
        # write
        for n in range(1, nbands + 1):
            dst_ds.GetRasterBand(n).WriteArray(masked_line[n - 1].astype(np.uint8),
                                               xoff=0, yoff=i)
    # close datasets
    source_ds, dst_ds = None, None


def get_iter_data(shapefile, batch_size=32, min_chip_hw=0, max_chip_hw=125,
                  classes=['No swimming pool', 'Swimming pool'], return_id = False,
                  normalize=True, img_name=None, return_labels=True, bit_depth=8,
                  image_id=None, show_percentage=True, mask=True, **kwargs):
    '''
    Generates batches of uniformly-sized training data from shapefile. If the shapefile has polygons from
        more than one image strip, strips must be named after their catalog id as it is
        referenced in the image_id property of each polyogn.

    INPUT   shapefile (string): name of shapefile to extract polygons from
            batch_size (int): number of chips to generate each iteration
            min_chip_hw (int): minimum size acceptable (in pixels) for a polygon.
                defaults to 10.
            max_chip_hw (int): maximum size acceptable (in pixels) for a polygon. Note
                that this will be the size of the height and width of input images to the
                net. defaults to 125.
            classes (list['string']): name of classes for chips. Defualts to swimming
                pool classes (['Swimming_pool', 'No_swimming_pool'])
            return_id (bool): return the geometry id with each chip. Defaults to False
            buffer (list[int]): two-dim buffer in pixels. defaults to [0,0].
            mask (bool): if True returns a masked array. defaults to True
            normalize (bool): divide all chips by max pixel intensity (normalize net
                input). Defualts to True.
            img_name (string): name of tif image to use for extracting chips. Defaults to
                None (the image name is assumed to be the image id listed in shapefile).
                This is only relevant if the shapefile has polygons only in one image
                strip. Otherwise each strip must be named after the image_id found in
                the polygon properties.
            return_labels (bool): Include labels in output. Defualts to True.
            bit_depth (int): Bit depth of the imagery, necessary for proper normalization.
                defualts to 8 (standard for dra'd imagery).
            image_id (string): Image id if you only want to generate data from a specific
                image strip. Defaults to None (will generate chips from all iamgery)
            show_percentage (bool): Print percent of chips collected to stdout. Defaults
                to True

    OUTPUT  Returns a generator object (g). calling g.next() returns the following:
            chips: one batch of masked (if True) chips
            corresponding feature_id for chips (if return_id is True)
            corresponding chip labels (if return_labels is True)

    EXAMPLE:
        >> g = get_iter_data('shapefile.geojson', batch-size=12)
        >> x,y = g.next()
        # x is the first 12 chips (of appropriate size) from the input shapefile
        # y is a list of classifications for the chips in x
    '''

    ct, inputs, labels, ids = 0, [], [], []
    nb_classes = len(classes)

    # determine which images to extract chips from
    if image_id:
        img_ids = [image_id]
    else:
        print 'Finding unique image ids...'
        img_ids = gt.find_unique_values(shapefile, property_name='image_id')

    # Create numerical class names
    cls_dict = {classes[i]: i for i in xrange(len(classes))}

    for img_id in img_ids:
        if not img_name:
            img = geoio.GeoImage(img_id + '.tif')
        else:
            img = geoio.GeoImage(img_name)

        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[{'image_id': img_id}],
                                                mask=mask, **kwargs):

            # check for adequate chip size
            if chip is None:
                if show_percentage:
                    sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch_size)) + ' ' * 5)
                    sys.stdout.flush()
                continue

            chan, h, w = np.shape(chip)
            pad_h, pad_w = max_chip_hw - h, max_chip_hw - w

            if min(h, w) < min_chip_hw or max(h, w) > max_chip_hw:
                if show_percentage:
                    sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch_size)) + ' ' * 5)
                    sys.stdout.flush()
                continue

            # zero-pad chip to standard net input size
            chip = chip.filled(0).astype(float)  # replace masked entries with zeros
            chip_patch = np.pad(chip, [(0, 0), (pad_h/2, (pad_h - pad_h/2)), (pad_w/2,
                (pad_w - pad_w/2))], 'constant', constant_values=0)

            if normalize:
                div = (2 ** bit_depth) - 1
                chip_patch /= float(div)

            # Get labels
            if return_labels:
                try:
                    label = properties['class_name']
                    if label is None:
                        continue
                    labels.append(cls_dict[label])
                except (TypeError, KeyError):
                    continue

            if return_id:
                id = properties['feature_id']
                ids.append(id)

            # do not include image_id for fitting net
            inputs.append(chip_patch)
            ct += 1
            if show_percentage:
                sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch_size)) + ' ' * 5)
                sys.stdout.flush()

            if ct == batch_size:
                data = [np.array([i for i in inputs])]

                if return_id:
                    data.append(ids)

                if return_labels:
                    # Create one-hot encoded labels
                    Y = np.zeros((batch_size, nb_classes))
                    for i in range(batch_size):
                        Y[i, labels[i]] = 1

                    data.append(Y)
                yield data
                ct, inputs, labels, ids = 0, [], [], []

    # return any remaining inputs
    if len(inputs) != 0:
        data = [np.array([i for i in inputs])]

        if return_id:
            data.append(ids)

        if return_labels:
            # Create one-hot encoded labels
            Y = np.zeros((len(labels), nb_classes))
            for i in range(len(labels)):
                Y[i, labels[i]] = 1
            data.append(Y)
        yield data


def get_data_from_polygon_list(features, min_chip_hw=0, max_chip_hw=125,
                               classes=['No swimming pool', 'Swimming pool'],
                               normalize=True, return_id=False, return_labels=True,
                               bit_depth=8, mask=True, show_percentage=True,
                               assert_all_valid=False, resize_dim=None, **kwargs):
    '''
    Returns pixel intensity array given a list of polygons (features) from an open geojson
        file. All chips woll be of usiform size. This enables extraction of pixel data
        multiple image strips using polygons not saved to disk. Will only return polygons
        of valid size (between min_chip_hw and max_chip_hw).

    Each image strip referenced in the image_id properties of
        polygons must be  in the working directory and named as follows: <image_id>.tif

    INPUTS  features (list): list of polygon features from an open geojson file.
                IMPORTANT: Geometries must be in the same projection as the imagery! No
                projection checking is done!
            min_chip_hw (int): minimum size acceptable (in pixels) for a polygon.
                defaults to 10.
            max_chip_hw (int): maximum size acceptable (in pixels) for a polygon. Note
                that this will be the size of the height and width of all output chips.
                defaults to 125.
            classes (list['string']): name of classes for chips. Defualts to swimming
                pool classes (['Swimming_pool', 'No_swimming_pool'])
            normalize (bool): divide all chips by max pixel intensity (normalize net
                input). Defualts to True.
            return_id (bool): return the feature id with each chip. Defaults to False.
            return_labels (bool): Include labels in output. Labels will be numerical
                and correspond to the class index within the classes argument. Defualts
                to True.
            bit_depth (int): Bit depth of the imagery, necessary for proper normalization.
            defualts to 8 (standard for dra'd imagery).
            show_percentage (bool): Print percent of chips collected to stdout. Defaults
                to True
            assert_all_valid (bool): Throw an error if any of the included polygons do not
                match the size criteria (defined by min and max_chip_hw), or are returned
                as None from geoio. Defaults to False.
            resize_dim (tup): Dimensions to reshape chips into after padding. Use for
                downsampling large chips. Dimensions: (n_chan, rows, cols). Defaults to
                None (does not resize).

            kwargs:
            -------
            bands (list of ints): The band numbers (base 1) to be retrieved from the
                imagery. Defualts to None (all bands retrieved)
            buffer (int or list of two ints): Number of pixels to add as a buffer around
                the requested pixels. If an int, the same number of pixels will be added
                to both dimensions. If a list of two ints, they will be interpreted as
                xpad and ypad.

    OUTPUT  (chips, labels). Chips will be of size (n_bands, max_chip_hw, max_chip_hw).
                Polygons will be zero-padded to the proper shape
    '''

    inputs, labels, ids = [], [], []
    ct, total = 0, len(features)
    nb_classes = len(classes)
    cls_dict = {classes[i]: i for i in xrange(len(classes))}
    imgs = {} # {image id: open image}


    # cycle through polygons and get pixel data
    for poly in features:
        id = poly['properties']['image_id']
        coords = poly['geometry']['coordinates'][0]

        # open all images in geoio
        if id not in imgs.keys():
            try:
                imgs[id] = geoio.GeoImage(id + '.tif')
            except (ValueError):
                raise Exception('{}.tif not found in current directory. Please make ' \
                                'sure all images refereced in features are present and ' \
                                'named properly'.format(str(id)))

        # call get_data on polygon geom
        chip = imgs[id].get_data_from_coords(coords, mask=mask, **kwargs)
        if chip is None:
            if show_percentage:
                sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(total)) + ' ' * 5)
                sys.stdout.flush()
            if assert_all_valid:
                raise Exception('Invalid polygon with feature id {}. Please make sure' \
                                'all polygons are valid or set assert_all_valid to ' \
                                'False.'.format(str(poly['properties']['image_id'])))
            continue

        # check for adequate chip size
        chan, h, w = np.shape(chip)
        pad_h, pad_w = max_chip_hw - h, max_chip_hw - w

        if min(h, w) < min_chip_hw or max(h, w) > max_chip_hw:
            if show_percentage:
                sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(total)) + ' ' * 5)
                sys.stdout.flush()
            if assert_all_valid:
                raise Exception('Polygon with feature id {} does not meet the size ' \
                                'requirements. Please filter the geojson first or ' \
                                'set assert_all_valid to False.')
            continue

        # zero-pad polygons to (n_bands, max_chip_hw, max_chip_hw)
        chip = chip.filled(0).astype(float)  # replace masked entries with zeros
        chip_patch = np.pad(chip, [(0, 0), (pad_h/2, (pad_h - pad_h/2)), (pad_w/2,
            (pad_w - pad_w/2))], 'constant', constant_values=0)

        # resize chip
        if resize_dim:
            chip_patch = np.resize(chip_patch, resize_dim)

        # norm pixel intenisty from 0 to 1
        if normalize:
            div = (2 ** bit_depth) - 1
            chip_patch /= float(div)

        # get labels
        if return_labels:
            try:
                label = poly['properties']['class_name']
                if label is None:
                    continue
                labels.append(cls_dict[label])
            except (TypeError, KeyError):
                continue

        # get feature ids
        if return_id:
            id = poly['properties']['feature_id']
            ids.append(id)

        # append chip to inputs
        inputs.append(chip_patch)
        ct += 1
        if show_percentage:
            sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(total)) + ' ' * 5)
            sys.stdout.flush()

    # combine data
    inputs = [np.array([i for i in inputs])]

    if return_id:
        inputs.append(ids)

    if return_labels:
        # format labels
        Y = np.zeros((len(labels), nb_classes))
        for i in range(len(labels)):
            Y[i, labels[i]] = 1
        inputs.append(Y)

    return inputs


class getIterData(object):
    '''
    A class for iteratively extracting chips from a geojson shapefile and one or more
        corresponding GeoTiff strips.
    INPUT   shapefile (string): name of shapefile to extract polygons from
            batch_size (int): number of chips to generate per call of self.create_batch().
                Defaults to 10000
            classes (list['string']): name of classes for chips. Defualts to swimming
                pool classes (['Swimming_pool', 'No_swimming_pool'])
            min_chip_hw (int): minimum size acceptable (in pixels) for a polygon.
                defaults to 30.
            max_chip_hw (int): maximum size acceptable (in pixels) for a polygon. Note
                that this will be the size of the height and width of input images to the
                net. defaults to 125.
            return_labels (bool): Include labels in output. Defualts to True.
            return_id (bool): return the geometry id with each chip. Defaults to False
            mask (bool): if True returns a masked array. defaults to True
            normalize (bool): divide all chips by max pixel intensity (normalize net
                input). Defualts to True.
            props (dict): Proportion of chips to extract from each image strip. If the
                proportions don't add to one they will each be divided by the total of
                the values. Defaults to None, in which case proportions will be
                representative of ratios in the shapefile.
            bit_depth (int): bit depth of the imagery, necessary for proper normalization.
                Defaults to 8.
    OUTPUT  creates a class instance that will produce batches of chips from the input
                shapefile when create_batch() is called.
    EXAMPLE
            $ data_generator = getIterData('shapefile.geojson', batch_size=1000)
            $ x, y = data_generator.create_batch()
            # x = batch of 1000 chips from all image strips
            # y = labels associated with x
    '''

    def __init__(self, shapefile, batch_size=10000, min_chip_hw=0, max_chip_hw=125,
                 classes=['No swimming pool', 'Swimming pool'], return_labels=True,
                 return_id=False, mask=True, normalize=True, props=None, bit_depth=8,
                 show_percentage=True, cycle=False):

        self.shapefile = shapefile
        self.batch_size = batch_size
        self.classes = classes
        self.min_chip_hw = min_chip_hw
        self.max_chip_hw = max_chip_hw
        self.return_labels = return_labels
        self.return_id = return_id
        self.mask = mask
        self.normalize = normalize
        self.bit_depth = bit_depth
        self.show_percentage = show_percentage

        # get image proportions
        print 'Getting image proportions...'
        if props:
            self.img_ids = props.keys()
            self.props = self._format_props_input(props)

        else:
            self.img_ids = gt.find_unique_values(shapefile, property_name='image_id')
            self.props = {}
            for id in self.img_ids:
                if np.around(self.get_proportion('image_id', id) * self.batch_size) > 0:
                    self.props[id] = int(np.around(self.get_proportion('image_id', id) * self.batch_size))

        # account for difference in batch size and total due to rounding
        total = np.sum(self.props.values())
        if total < batch_size:
            diff = np.random.choice(self.props.keys())
            self.props[diff] += (batch_size - total)

        if total > batch_size:
            diff = max(self.props.iterkeys(), key=(lambda key: self.props[key]))[0]
            self.props[diff] -= (total - batch_size)

        # initialize generators
        print 'Creating chip generators...'
        self.chip_gens = {}
        for id in self.props.keys():
            if cycle:
                self.chip_gens[id] = self.yield_infinitely_from_img_id(id,
                                                                     batch=self.props[id])
            else:
                self.chip_gens[id] = self.yield_from_img_id(id, batch=self.props[id])

    def _format_props_input(self, props):
        '''
        helper function to format the props dict input
        '''
        # make sure proportions add to one
        total_prop = np.sum(props.values())
        for i in props.keys():
            props[i] /= float(total_prop)

        p_new = {i: int(props[i] * self.batch_size) for i in props.keys()}
        return p_new

    def get_proportion(self, property_name, property):
        '''
        Helper function to get the proportion of polygons with a given property in a
            shapefile
        INPUT   shapefile (string): name of the shapefile containing the polygons
                property_name (string): name of the property to search for exactly as it
                    appears in the shapefile properties (ex: image_id)
                property (string): the property of which to get the proportion of in the
                    shapefile (ex: '1040010014800C00')
        OUTPUT  proportion (float): proportion of polygons that have the property of interest
        '''
        total, prop = 0,0

        # open shapefile, get polygons
        with open(self.shapefile) as f:
            data = geojson.load(f)['features']

        # loop through features, find property count
        for polygon in data:
            total += 1

            try:
                if str(polygon['properties'][property_name]) == property:
                    prop += 1
            except (KeyError):
                print 'warning: not all polygons have property ' + str(property_name)
                continue

        return float(prop) / total

    def yield_from_img_id(self, img_id, batch):
        '''
        helper function to yield data from a given shapefile for a specific img_id. This
        function should ONLY be use with the keras fit_generator function
        INPUT   img_id (str): ids of the images from which to generate patches from
                batch (int): number of chips to generate per iteration from the input
                    image id
        OUTPUT  Returns a generator object (g). calling g.next() returns the following:
                chips:
                    - one batch of masked (if True) chips
                    - corresponding feature_id for chips (if return_id is True)
                    - corresponding chip labels (if return_labels is True)
        EXAMPLE:
            $ g = get_iter_data('shapefile.geojson', batch-size=12)
            $ x,y = g.next()
            # x is the first 12 chips (of appropriate size) from the input shapefile
            # y is a list of classifications for the chips in x
        '''

        ct, inputs, labels, ids = 0, [], [], []
        cls_dict = {self.classes[i]: i for i in xrange(len(self.classes))}

        img = geoio.GeoImage(img_id + '.tif')
        for chip, properties in img.iter_vector(vector=self.shapefile,
                                                properties=True,
                                                filter=[{'image_id': img_id}],
                                                mask=self.mask):
            # check for adequate chip size
            if chip is None:
                if self.show_percentage:
                    sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch)) + ' ' * 5)
                    sys.stdout.flush()
                continue

            chan, h, w = np.shape(chip)
            pad_h, pad_w = self.max_chip_hw - h, self.max_chip_hw - w
            if min(h, w) < self.min_chip_hw or max(h, w) > self.max_chip_hw:
                if self.show_percentage:
                    sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch)) + ' ' * 5)
                    sys.stdout.flush()
                continue

            # zero-pad chip to standard net input size
            chip = chip.filled(0).astype(float)  # replace masked entries with zeros
            chip_patch = np.pad(chip, [(0, 0), (pad_h/2, (pad_h - pad_h/2)), (pad_w/2,
                (pad_w - pad_w/2))], 'constant', constant_values=0)

            if self.normalize:
                div = (2 ** self.bit_depth) - 1
                chip_patch /= float(div)

            # get labels
            if self.return_labels:
                try:
                    label = properties['class_name']
                    if label is None:
                        continue
                    labels.append(cls_dict[label])
                except (TypeError, KeyError):
                    continue

            # get id
            if self.return_id:
                id = properties['feature_id']
                ids.append(id)

            inputs.append(chip_patch)
            ct += 1
            if self.show_percentage:
                sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch)) + ' ' * 5)
                sys.stdout.flush()

            if ct == batch:
                data = [np.array([i for i in inputs])]

                if self.return_id:
                    data.append(ids)

                # Create one-hot encoded labels
                if self.return_labels:
                    Y = np.zeros((batch, len(self.classes)))
                    for i in range(batch):
                        Y[i, labels[i]] = 1
                    data.append(Y)
                yield data
                ct, inputs, labels, ids = 0, [], [], []

    def yield_infinitely_from_img_id(self, img_id, batch):
        '''
        Same as yield_from_img_id except this function will loop infinitely though the
        data. This may result in duplicate data in each batch, or infinite loops. The
        function should therefore ONLY be use with the keras fit_generator function.
        INPUT   img_id (str): ids of the images from which to generate patches from
                batch (int): number of chips to generate per iteration from the input
                    image id
        OUTPUT  Returns a generator object (g). calling g.next() returns the following:
                chips:
                    - one batch of masked (if True) chips
                    - corresponding feature_id for chips (if return_id is True)
                    - corresponding chip labels (if return_labels is True)
        EXAMPLE:
            $ g = get_iter_data('shapefile.geojson', batch-size=12)
            $ x,y = g.next()
            # x is the first 12 chips (of appropriate size) from the input shapefile
            # y is a list of classifications for the chips in x
        '''

        ct, inputs, labels, ids = 0, [], [], []
        cls_dict = {self.classes[i]: i for i in xrange(len(self.classes))}

        img = geoio.GeoImage(img_id + '.tif')
        for chip, properties in cycle(img.iter_vector(vector=self.shapefile,
                                                      properties=True,
                                                      filter=[{'image_id': img_id}],
                                                      mask=self.mask)):
            # check for adequate chip size
            if chip is None:
                if self.show_percentage:
                    sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch)) + ' ' * 5)
                    sys.stdout.flush()
                continue

            chan, h, w = np.shape(chip)
            pad_h, pad_w = self.max_chip_hw - h, self.max_chip_hw - w
            if min(h, w) < self.min_chip_hw or max(h, w) > self.max_chip_hw:
                if self.show_percentage:
                    sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch)) + ' ' * 5)
                    sys.stdout.flush()
                continue

            # zero-pad chip to standard net input size
            chip = chip.filled(0).astype(float)  # replace masked entries with zeros
            chip_patch = np.pad(chip, [(0, 0), (pad_h/2, (pad_h - pad_h/2)), (pad_w/2,
                (pad_w - pad_w/2))], 'constant', constant_values=0)

            if self.normalize:
                div = (2 ** self.bit_depth) - 1
                chip_patch /= float(div)

            # get labels
            if self.return_labels:
                try:
                    label = properties['class_name']
                    if label is None:
                        continue
                    labels.append(cls_dict[label])
                except (TypeError, KeyError):
                    continue

            # get id
            if self.return_id:
                id = properties['feature_id']
                ids.append(id)

            inputs.append(chip_patch)
            ct += 1
            if self.show_percentage:
                sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch)) + ' ' * 5)
                sys.stdout.flush()

            if ct == batch:
                data = [np.array([i for i in inputs])]

                if self.return_id:
                    data.append(ids)

                # Create one-hot encoded labels
                if self.return_labels:
                    Y = np.zeros((batch, len(self.classes)))
                    for i in range(batch):
                        Y[i, labels[i]] = 1
                    data.append(Y)
                yield data
                ct, inputs, labels, ids = 0, [], [], []

    def next(self):
        '''
        generate a batch of chips
        '''
        data = []

        # hit each generator in chip_gens
        for img_id, gen in self.chip_gens.iteritems():
            if self.show_percentage:
                print '\nCollecting chips for image ' + str(img_id) + '...'
            try:
                data += zip(*gen.next())
            except (StopIteration):
                return 'Not enough polygon data, please use a smaller batch size'

        np.random.shuffle(data)
        return [np.array(i) for i in zip(*data)]
