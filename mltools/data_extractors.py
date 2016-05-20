# Extract pixels and metadata using shapefiles and georeferenced imagery.
# The purpose of this module is to generate train, test and target data
# for machine learning algorithms.

import geoio
import geojson_tools as gt
import numpy as np
import osgeo.gdal as gdal
from osgeo.gdalconst import *


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
        masked_line = line*(mask_line>0)
        # write 
        for n in range(1, nbands+1):
            dst_ds.GetRasterBand(n).WriteArray(masked_line[n-1].astype(np.uint8), 
                                               xoff=0, yoff=i)       
    # close datasets
    source_ds, dst_ds = None, None
