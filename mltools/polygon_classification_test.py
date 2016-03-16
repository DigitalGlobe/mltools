# Test PolygonClassifier on a set of classified polygons.
# The script pulls a set of classified polygons from Tomnod, splits it
# in train and test set and evaluates the confusion matrix. 
# This script was used in the adelaide_pools_2016 campaign. 

import json
import json_tools as jt
import os
import sys

from polygon_classifier import PolygonClassifier
from crowdsourcing import TomnodCommunicator

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

# define, train and test the classifier
c = PolygonClassifier(algorithm_params)
print 'Train classifier'
c.train(train_filename)
print 'Test classifier'
labels, scores, C = c.classify(test_filename, return_confusion_matrix=True)

print 'Confusion matrix:'
print C
print 'Normalized confusion matrix:'
print C.astype(float)/C.sum(1)[:, None]

