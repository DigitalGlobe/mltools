# Contains functions for manipulating jsons and geojsons.

import geojson

from shapely.wkb import loads


def join_geojsons(filenames, output_file):
    """Join geojsons into one. The spatial reference system of the 
       output file is the same as the one of the last file in the list.

       Args:
           filenames: List of filenames (have to be geojson).
           output_file (str): Output filename (has to be geojson).
    """

    # get feature collections
    final_features  = []
    for file in filenames:
        with open(file) as f:
           feat_collection = geojson.load(f)
           final_features += feat_collection['features']

    feat_collection['features'] = final_features

    # write to output file
    with open(output_file, 'w') as f:
        geojson.dump(feat_collection, f) 

    
def split_geojson(input_file, file_1, file_2, no_in_first_file):
    """Split a geojson in two separate files.
       
       Args:
           input_file (str): Input filename (ext. geojson).
           file_1 (str): Output filename 1 (ext. geojson).
           file_2 (str): Output filename 2 (ext. geojson).
           no_features (int): Number of features in input_file to go to file_1.
           output_file (str): Output filename (ext. geojson).
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


def get_from_geojson(filename, property_names):
    """Reads a geojson and returns a list of value tuples, each value
       corresponding to a property in property_names.

       Args:
           filename (str): File name (has to be geojson).
           property_names: List of strings. Each string is a property name.

       Returns:
           List of value tuples.     
    """

    # get feature collections
    with open(filename) as f:
        feature_collection = geojson.load(f)

    features = feature_collection['features']
    values = [tuple([feat['properties'].get(x) 
                    for x in property_names]) for feat in features]

    return values    


def write_to_geojson(data, property_names, output_file):
    '''Write list of tuples to geojson. 
       First entry of each tuple should be geometry in hex coordinates 
       and the rest properties.

       Args:
           data: List of tuples.
           property_names: List of strings. Should be same length as each 
                           tuple in data.
           output_file (str): File to write to (should be .geojson).
                           
    '''        

    geojson_features = [] 
    for entry in data:
        coords_in_hex, properties = entry[0], entry[1:]
        polygon = loads(coords_in_hex, hex=True)
        coords = [list(polygon.exterior.coords)]   # the brackets are dictated
                                                   # by geojson format!!! 
        property_dict = dict(zip(property_names, properties))
        geojson_feature = geojson.Feature(geometry=geojson.Polygon(coords), 
                                          properties=property_dict)
        geojson_features.append(geojson_feature)

    feature_collection = geojson.FeatureCollection(geojson_features)    

    with open(output_file, 'wb') as f:
        geojson.dump(feature_collection, f)


def write_properties_to_geojson(data, property_names, input_file, output_file):
    """Writes property data to polygon_file to create output_file.
       The length of data must be equal to the number of features in 
       input_file. If some of the features in polygon_file already have 
       values for the corresponding properties, the values are overwritten.

       Args:
           data (list): List of tuples. Each entry is a tuple of dimension equal  
                        to property_names. 
           property_names (list): Property names.
           input_file (str): Input filename (has to be .geojson)
           output_file (str): Output filename (has to be .geojson)
    """

    with open(input_file) as f:
        feature_collection = geojson.load(f)

    features = feature_collection['features']

    for i, feature in enumerate(features):
        for j, property_value in enumerate(data[i]):
            feature['properties'][property_names[j]] = property_value

    feature_collection['features'] = features    

    with open(output_file, 'w') as f:
        geojson.dump(feature_collection, f)     
  