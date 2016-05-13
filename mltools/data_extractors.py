# Extract pixels and metadata using shapefiles and georeferenced imagery.
# The purpose of this module is to generate train, test and target data
# for machine learning algorithms.


import geoio
import geojson_tools as gt


def get_data(shapefile, return_labels=False, buffer=[0,0]):
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
                                                buffer=buffer):
            
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


def sliding_window(image, chip_size, stride, max_chips = 10000):
    """Implement a sliding window on a georeferenced image.
       
       Args:
           image (str): Image filename.
           chip_size (list): Array of chip dimensions.
           stride (list): Window stride.
       
       Returns:
           List of chip rasters.
    """   
    img = geoio.GeoImage(image)
    
    # comprehension doesn't work?
    #chips = [chip for i, chip in enumerate(img.iter_window(win_size=chip_size, stride=stride)) if i<= max_chips-1]

    chips = []
    for i, chip in enumerate(img.iter_window(win_size=chip_size, stride=stride)):
        chips.append(chip)
        if i == max_chips-1: break

    return chips

# TO DO --- geoio probably needs to implement this
#def get_random_chips(image, no_chips, chip_size):
#    """Return randomly selected chips from georeferenced image.
#     
#       Args:
#           image (str): Image filename.
#           no_chips (int): Number of chips.
#           chip_size (list): Array of chip dimensions.
#       
#       Returns:
#          List of chip rasters.
#    """
#    img = geoio.GeoImage(image)
#    chips = []
    
    
  
 
