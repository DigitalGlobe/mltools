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
    final_features  = []
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
           property_names: List of strings. Should be same length as each tuple in data.
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


def write_properties_to(data, property_names, input_file, output_file, filter=None):
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
    values = np.array([ feat['properties'].get(property_name) for feat in features])
    
    return np.unique(values)
