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


def get_data(shapefile, return_labels=False, buffer=[0, 0], mask=False):
    """Return pixel intensity array for each geometry in shapefile.
       The image reference for each geometry is found in the image_id
       property of the shapefile.
       If shapefile contains points, then buffer must have non-zero entries.
       The function also returns a list of geometry ids; this is useful in
       case some of the shapefile entries do not produce a valid intensity
       array and/or class name.

       Args:
           shapefile (str): Name of shapefile in mltools geojson format.
           return_labels (bool): If True, then a label vector is returned.
           buffer (list): 2-dim buffer in PIXELS. The size of the box in each
                          dimension is TWICE the buffer size.
           mask (bool): Return a masked array.

       Returns:
           chips (list): List of pixel intensity numpy arrays.
           ids (list): List of corresponding geometry ids.
           labels (list): List of class names, if return_labels=True
    """

    data = []

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

    return zip(*data)


def get_iter_data(shapefile, batch_size=32, min_chip_hw=0, max_chip_hw=125,
                  classes=['No swimming pool', 'Swimming pool'], return_id = False,
                  normalize=True, img_name=None, return_labels=True, bit_depth=8,
                  image_id=None, show_percentage=True, mask=True, **kwargs):
    '''
    Generates batches of training data from shapefile. If the shapefile has polygons from
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

            # # resize image
            # if resize_dim:
            #     if resize_dim != chip_patch.shape:
            #         chip_patch = resize(chip_patch, resize_dim)

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
                               assert_all_valid=False, **kwargs):
    '''
    Returns pixel intensity array given a list of polygons (features) from an open geojson
        file. This enables extraction of pixel data multiple image strips using polygons
        not saved to disk. Will only return polygons of valid size (between min_chip_hw
        and max_chip_hw).

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
            return_labels (bool): Include labels in output. Defualts to True.
            bit_depth (int): Bit depth of the imagery, necessary for proper normalization.
            defualts to 8 (standard for dra'd imagery).
            show_percentage (bool): Print percent of chips collected to stdout. Defaults
                to True
            assert_all_valid (bool): Throw an error if any of the included polygons do not
                match the size criteria (defined by min and max_chip_hw), or are returned
                as None from geoio. Defaults to False.

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
    data = [np.array([i for i in inputs])]

    if return_id:
        data.append(ids)

    if return_labels:
        # format labels
        Y = np.zeros((len(labels), nb_classes))
        for i in range(len(labels)):
            Y[i, labels[i]] = 1
        data.append(Y)

    return data


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
