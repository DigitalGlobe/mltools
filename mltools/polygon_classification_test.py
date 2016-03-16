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
max_area = job['max_area']                            # max polygon area in m2
ratio_train_to_test = job['ratio_train_to_test']   
algorithm_params = job['algorithm_params']


# get tomnod credentials
with open('credentials.json', 'r') as f:
    credentials = json.load(f)

# initialize Tomnod communicator class
tc = TomnodCommunicator(credentials)

# fetch high confidence features and separate in train/test for each class
train_filenames, test_filenames = [], []
ratio_train_to_total = float(ratio_train_to_test)/(ratio_train_to_test + 1)
for i, class_entry in enumerate(classes):
    class_name = class_entry['name']
    no_samples = class_entry['no_samples']    
    print 'Collect', str(no_samples), 'samples for class', class_name

    gt_filename = '_'.join([class_name, catalog_id, 'gt.geojson'])
    data = tc.get_high_confidence_features(campaign_schema = schema, 
                                           image_id = catalog_id, 
                                           class_name = class_name,
                                           max_number = no_samples,
                                           write_to = gt_filename)

    train_filenames.append('_'.join([class_name, catalog_id, 'train.geojson']))
    test_filenames.append('_'.join([class_name, catalog_id, 'test.geojson']))
    jt.split_geojson(gt_filename, 
                     train_filenames[i], 
                     test_filenames[i], 
                     no_in_first_file = int(no_samples*ratio_train_to_total))
    

# assemble train and test files by joining constituent train and test files
train_filename = '_'.join([catalog_id, 'train.geojson'])
test_filename = '_'.join([catalog_id, 'test.geojson'])
jt.join_geojsons(train_filenames, train_filename)
jt.join_geojsons(test_filenames, test_filename)


# define, train and test the classifier
c = PolygonClassifier(algorithm_params)
c.train(train_filename)
labels, scores, C = c.classify(test_filename, return_confusion_matrix=True)

print C

