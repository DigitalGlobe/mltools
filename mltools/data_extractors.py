# Extract pixels and metadata using shapefiles and georeferenced imagery.
# The purpose of this module is to generate train, test and target data
# for machine learning algorithms.

from itertools import cycle
import geoio
import geojson_tools as gt


def get_data(shapefile, return_labels=False, buffer=[0,0], mask=False):
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
                                                filter=[{'image_id':image_id}],
                                                buffer=buffer,
                                                mask=mask):

            if chip is None or reduce(lambda x, y: x*y, chip.shape)==0:
                continue

            this_data = [chip, properties['feature_id']] # every geometry must have id

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


def get_iter_data(shapefile, batch_size=32, nb_classes=2, min_chip_hw=100, max_chip_hw=224, return_labels=True, buffer=[0,0], mask=True, fc=False, resize_dim=None):
    '''
    Generates batches of training data from shapefile for when it will not fit in memory.

    INPUT   (1) string 'shapefile': name of shapefile to extract polygons from
            (2) int 'batch_size': number of chips to generate per iteration. equal to batch-size of net, defaults to 32
            (3) int 'nb_classes': number of classes in which to categorize itmes
            (4) int 'min_chip_hw': minimum size acceptable (in pixels) for a polygon. defaults to 100
            (5) int 'max_chip_hw': maximum size acceptable (in pixels) for a polygon. note that this will be the size of the height and width of input images to the net (default = 224)
            (6) bool 'return_labels': return class label with chips. defaults to True
            (7) list[int] 'buffer': two-dim buffer in pixels. defaults to [0,0].
            (8) bool 'mask': if True returns a masked array. defaults to True
            (9) bool 'fc': return appropriately shaped target vector for FCNN
            (10) tuple(int) 'resize': size to downsample chips to (channels, height, width). Note that resizing takes place after padding the original polygon. Defaults to None (do not resize).

    OUTPUT  (1) chips: one batch of masked (if True) chips
            (2) corresponding feature_id for chips
            (3) corresponding chip labels (if True)
    '''

    ct, inputs, labels = 0, [], []
    print 'Extracting image ids...'
    img_ids = gt.find_unique_values(shapefile, property_name='image_id')

    for img_id in cycle(img_ids):
        img = geoio.GeoImage(img_id + '.tif')

        for chip, properties in img.iter_vector(vector=shapefile,
                                                properties=True,
                                                filter=[{'image_id':img_id}],
                                                buffer=buffer,
                                                mask=mask):

            # check for adequate chip size
            chan, h, w = np.shape(chip)
            if chip is None or min(h, w) < min_chip_hw or max(h, w) > max_chip_hw:
                continue

            # zero-pad chip to standard net input size
            chip = chip.filled(0)[:3] # replace masked entries with zeros
            chip_patch = np.pad(chip, [(0,0), (0, max_chip_hw - h), (0, max_chip_hw - w)], 'constant', constant_values = 0)

            # resize image
            if resize_dim != chip_patch.shape:
                chip_patch = resize(chip_patch, resize_dim)

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

    # # return any remaining inputs
    # if len(inputs) != 0:
    #     l = [1 if lab == 'Swimming pool' else 0 for lab in labels]
    #     labels = np_utils.to_categorical(l, 2)
    #     yield (np.array([i[:3] for i  in inputs]), np.array(labels))

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
    for i, chip in enumerate(img.iter_window_random(win_size=chip_size, no_chips=no_chips)):
        chips.append(chip)
        if i == no_chips-1: break

    return chips
