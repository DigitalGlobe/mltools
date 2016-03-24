# Test PolygonClassifier on a set of classified polygons.
# The script pulls a set of classified polygons from Tomnod, splits it
# in train and test set and evaluates the confusion matrix. 
# This script was used in the adelaide_pools_2016 campaign. 

import json
import os
import sys

from mltools import features
from mltools import json_tools as jt
from mltools.polygon_classifier import PolygonClassifier
from mltools.crowdsourcing import TomnodCommunicator

# suppress annoying warnings
import warnings
warnings.filterwarnings("ignore")

# get job parameters
with open('job.json', 'r') as f:
    job = json.load(f)
    
schema = job['schema']
catalog_id = job['catalog_id']
classes = job['classes']
algorithm_params = job['algorithm_params']


# get tomnod credentials
with open('credentials.json', 'r') as f:
    credentials = json.load(f)

# initialize Tomnod communicator class
tc = TomnodCommunicator(credentials)

# fetch high confidence features and separate in train/test for each class
train_filenames, test_filenames = [], []
for i, class_entry in enumerate(classes):
    class_name = class_entry['name']
    no_samples = class_entry['no_samples']    
    no_train_samples = class_entry['no_train_samples']
    min_votes = class_entry['min_votes']
    max_area = class_entry['max_area']
    print 'Collect {} {} samples from schema {} and image {}'.format(no_samples, 
                                                                     class_name, 
                                                                     schema, 
                                                                     catalog_id) 

    gt_filename = '_'.join([class_name, catalog_id, 'gt.geojson'])
    data = tc.get_high_confidence_features(campaign_schema = schema, 
                                           image_id = catalog_id, 
                                           class_name = class_name,
                                           max_number = no_samples,
                                           min_votes = min_votes,
                                           max_area = max_area)
    jt.write_to_geojson(data = data,
                        property_names = ['feature_id', 'image_name', 'class_name'],
                        output_file = gt_filename)

    train_filenames.append('_'.join([class_name, catalog_id, 'train.geojson']))
    test_filenames.append('_'.join([class_name, catalog_id, 'test.geojson']))
    jt.split_geojson(gt_filename, 
                     train_filenames[i], 
                     test_filenames[i], 
                     no_in_first_file = no_train_samples)
    

# assemble train and test files by joining constituent train and test files
train_filename = '_'.join([catalog_id, 'train.geojson'])
test_filename = '_'.join([catalog_id, 'test.geojson'])
jt.join_geojsons(train_filenames, train_filename)
jt.join_geojsons(test_filenames, test_filename)

# instantiate polygon classifier
c = PolygonClassifier(algorithm_params)

# override default feature extraction method
def feature_extractor(data):
    '''Feature extractor for swimming pool detection.
       Args:
           data (numpy array): Pixel data vector.          
       Returns:
           Feature numpy vector.
    '''

    # hard-coded
    pool_sig = np.array([1179, 2295, 2179, 759, 628, 186, 270, 110])
    covered_pool_sig = np.array([1584, 1808, 1150, 1104, 1035, 995, 1659, 1741])
    
    pool_data = features.spectral_angles(data, pool_sig)
    covered_pool_data = features.spectral_angles(data, covered_pool_sig)
    band26_ratio = features.band_ratios(data, 2, 6)
    band36_ratio = features.band_ratios(data, 3, 6)

    return [np.max(band26_ratio), np.max(band36_ratio), np.min(pool_data), np.min(covered_pool_data)]

c.feature_extractor = feature_extractor

print 'Train classifier'
c.train(train_filename)

print 'Test classifier'
labels, scores, C = c.classify(test_filename, return_confusion_matrix=True)

print 'Confusion matrix:'
print C
print 'Normalized confusion matrix:'
print C.astype(float)/C.sum(1)[:, None]