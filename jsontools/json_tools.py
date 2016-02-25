"""
Contains functions for manipulating jsons and geojsons.

Author: Kostas Stamatiou
Created: 02/24/2016
Contact: kostas.stamatiou@digitalglobe.com
"""

import geojson

from sklearn.metrics import confusion_matrix


def join_two_geojsons(file_1, file_2, output_file):
	"""Join two geojsons into one. The spatial reference system of the 
	   output file is the same as the one of file_1.

	   Args:
	       file_1 (str): Filename 1 (ext. geojson).
	       file_2 (str): Filename 2 (ext. geojson).
	       output_file (str): Output filename (ext. geojson).
	"""

	# get feature collections
	with open(file_1) as f:
	    feat_collection_1 = geojson.load(f)

	with open(file_2) as f:
	    feat_collection_2 = geojson.load(f)

	feat_final = feat_collection_1['features'] + feat_collection_2['features']  

	feat_collection_1['features'] = feat_final

	# write to output file
	with open(output_file, 'w') as f:
	    geojson.dump(feat_collection_1, f) 

    
def split_geojson(input_file, file_1, file_2, ratio):
	"""Split a geojson in two separate files.
	   
	   Args:
	       input_file (str): Input filename (ext. geojson).
	       file_1 (str): Output filename 1 (ext. geojson).
	       file_2 (str): Output filename 2 (ext. geojson).
	       ratio (float): Proportion of features in input_file that goes to 
	                      file_1. ratio is from 0 to 1.
	       output_file (str): Output filename (ext. geojson).
	"""

	# get feature collection
	with open(input_file) as f:
	    feat_collection = geojson.load(f)

	features = feat_collection['features']
	no_features = len(features)
	no_features_1 = int(round(ratio*no_features))
	feat_collection_1 = geojson.FeatureCollection(features[0:no_features_1])
	feat_collection_2 = geojson.FeatureCollection(features[no_features_1:])

	with open(file_1, 'w') as f:
	    geojson.dump(feat_collection_1, f) 

	with open(file_2, 'w') as f:
		geojson.dump(feat_collection_2, f) 	


def get_classes_from_geojson(input_file):
	"""Reads a geojson with class_name property and returns a vector
	   with all the class names.

	   Args:
	       input_file (str): Input filename (.geojson extension).

	   Returns:
	       A list of class names (list).     
	"""

	# get feature collections
	with open(input_file) as f:
	    feature_collection = geojson.load(f)

	features = feature_collection['features']    
	labels = [feat['properties']['class_name'] for feat in features]

	return labels    


def confusion_matrix_two_geojsons(file_1, file_2, class_names):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.

	Args:
	    file_1 (str): Ground truth filename (.geojson extension)
	    file_2 (str): Prediction filename (.geojson extension)
	    class_names (list): List of class names. Each class name is a string.

	Returns:
	    An integer numpy array C, where C[i,j] is the number of observations 
	    known to be in group i but predicted to be in group j (numpy array).     
	"""

	true_classes = get_classes_from_geojson(file_1)
	pred_classes = get_classes_from_geojson(file_2)

	C = confusion_matrix(true_classes, pred_classes, class_names)

	return C

