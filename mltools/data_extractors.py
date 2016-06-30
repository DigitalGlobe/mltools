# Extract pixels and metadata using shapefiles and georeferenced imagery.
# The purpose of this module is to generate train, test and target data
# for machine learning algorithms.

import geoio
import geojson_tools as gt
import numpy as np
import sys
import osgeo.gdal as gdal
from osgeo.gdalconst import *
from functools import reduce
from sklearn.preprocessing import OneHotEncoder

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


def get_iter_data(shapefile, batch_size=32, nb_classes=2, min_chip_hw=30, max_chip_hw=125,
                  classes=['Swimming pool', 'No swimming pool'], return_id = False,
                  buffer=[0, 0], mask=True, resize_dim=None, normalize=True,
                  img_name=None):
    '''
    Generates batches of training data from shapefile. Labeles will be one-hot encoded.

    INPUT   shapefile (string): name of shapefile to extract polygons from
            batch_size (int): number of chips to generate each iteration
            nb_classes (int): number of classes in which to categorize itmes
            min_chip_hw (int): minimum size acceptable (in pixels) for a polygon.
                defaults to 30.
            max_chip_hw (int): maximum size acceptable (in pixels) for a polygon. Note
                that this will be the size of the height and width of input images to the
                net. defaults to 125.
            classes (list['string']): name of classes for chips. Defualts to swimming
                pool classes (['Swimming_pool', 'No_swimming_pool'])
            return_id (bool): return the geometry id with each chip. Defaults to False
            buffer (list[int]): two-dim buffer in pixels. defaults to [0,0].
            mask (bool): if True returns a masked array. defaults to True
            resize_dim (tuple(int)): size to downsample chips to (channels, height,
                width). Note that resizing takes place after padding the original polygon.
                Defaults to None (do not resize).
            normalize (bool): divide all chips by max pixel intensity (normalize net
                input)
            img_name (string): optional- name of tif image to use for extracting chips

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
    print 'Extracting image ids...'
    img_ids = gt.find_unique_values(shapefile, property_name='image_id')

    # Create numerical class names
    cls_dict = {classes[i]: i for i in xrange(len(classes))}

    for img_id in img_ids:
        if not img_name:
            img = geoio.GeoImage(img_id + '.tif')
        else:
            img = img_name

        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[{'image_id': img_id}],
                                                buffer=buffer,
                                                mask=mask):

            # check for adequate chip size
            chan, h, w = np.shape(chip)
            pad_h, pad_w = max_chip_hw - h, max_chip_hw - w
            if chip is None or min(h, w) < min_chip_hw or max(
                    h, w) > max_chip_hw:
                continue

            # zero-pad chip to standard net input size
            chip = chip.filled(0).astype(float)  # replace masked entries with zeros
            chip_patch = np.pad(chip, [(0, 0), (pad_h/2, (pad_h - pad_h/2)), (pad_w/2, (pad_w - pad_w/2))], 'constant', constant_values=0)

            # resize image
            if resize_dim:
                if resize_dim != chip_patch.shape:
                    chip_patch = resize(chip_patch, resize_dim)

            if normalize:
                chip_patch /= 255.

            # Get labels
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
            sys.stdout.write('\r%{0:.2f}'.format(100 * ct / float(batch_size)) + ' ' * 5)
            sys.stdout.flush()

            if ct == batch_size:

                # Create one-hot encoded labels
                Y = np.zeros((batch_size, nb_classes))
                for i in range(batch_size):
                    Y[i, labels[i]] = 1

                if return_id:
                    yield (np.array([i for i in inputs]), ids, Y)
                else:
                    yield (np.array([i for i in inputs]), Y)
                ct, inputs, labels, ids = 0, [], [], []

    # return any remaining inputs
    if len(inputs) != 0:

        # Create one-hot encoded labels
        Y = np.zeros((len(labels), nb_classes))
        for i in range(len(labels)):
            Y[i, labels[i]] = 1

        if return_id:
            yield (np.array([i for i in inputs]), ids, Y)
        else:
            yield (np.array([i for i in inputs]), Y)


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
