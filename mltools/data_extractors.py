# Extract pixels and metadata from georeferenced imagery.

import geoio
import geojson_tools as gt

def extract(polygon_file, return_class=False):
    """Extracts pixel intensities and class name for each polygon in polygon_file.
       The image reference for each polygon is found in the image_id
       property of the polygon_file.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               mltools accepted geojson format.
           return_class (bool): If True, then the iterator yields pixel
                                intensities and class name. If False, 
                                the iterator just yields pixel intensities.   
       
       Yields:
           Pixel intensities as masked numpy array and class name, 
           if return_class is True. 
           The iterator skips the entries of polygon_file where the pixels 
           can not be extracted. 
           If return_class=True, the iterator skips the entries of polygon_file
           where the class_name does not exist.
    """

    # go through polygon_file and unique image_id's
    image_ids = gt.find_unique_values(polygon_file, property_name='image_id')

    for image_id in image_ids:

        # add tif extension
        img = geoio.GeoImage(image_id + '.tif')

        if return_class:
             
            for this_raster, this_class in img.iter_vector(vector=polygon_file, 
                                                           properties='class_name', 
                                                           filter=[{'image_id':image_id}]):   
                if this_raster is None or this_class is None:
                    continue
                if this_class is not None:    # iter_vector returns {'class_name':'blah'}
                    this_class = this_class['class_name']
                yield this_raster, this_class
             
        else:

            for this_raster in img.iter_vector(vector=polygon_file,
                                               filter=[{'image_id':image_id}]):
                if this_raster is None:
                    continue
                yield this_raster                                                    
