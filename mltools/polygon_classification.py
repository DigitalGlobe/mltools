# Deploy PolygonClassifier on a set of unclassified polygons.
# The script pulls a set of classified polygons from Tomnod for training
# and a set of unclassified polygons for classification. 
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
target_params = job['target_params']
algorithm_params = job['algorithm_params']


# get tomnod credentials
with open('credentials.json', 'r') as f:
    credentials = json.load(f)

# initialize Tomnod communicator class
tc = TomnodCommunicator(credentials)

# fetch high confidence features for training
train_filenames = []
for i, class_entry in enumerate(classes):
    class_name = class_entry['name']
    no_train_samples = class_entry['no_train_samples']
    min_votes = class_entry['min_votes']
    max_area = class_entry['max_area']    
    print 'Collect {} {} samples from schema {} and image {}'.format(no_train_samples,
                                                                     class_name,
                                                                     schema,
                                                                     catalog_id)
    train_filenames.append('_'.join([class_name, catalog_id, 'train.geojson']))
    data = tc.get_high_confidence_features(campaign_schema = schema, 
                                           image_id = catalog_id, 
                                           class_name = class_name,
                                           max_number = no_train_samples,
                                           min_votes = min_votes,
                                           max_area = max_area)
    jt.write_to_geojson(data = data, 
                        property_names = ['feature_id', 'image_name', 'class_name'],
                        output_file = train_filenames[i])
    
# assemble final train file by joining constituent train files
train_filename = '_'.join([catalog_id, 'train.geojson'])
jt.join_geojsons(train_filenames, train_filename)

# fetch unclassified features for classification
target_filename = '_'.join([catalog_id, 'target.geojson'])
max_polys_to_classify = target_params['max_polys_to_classify']
max_area = target_params['max_area']
max_votes = target_params['max_votes'] 
data = tc.get_low_confidence_features(campaign_schema = schema, 
                                      image_id = catalog_id, 
                                      max_number = max_polys_to_classify,
                                      max_area = max_area,
                                      max_votes = max_votes)
jt.write_to_geojson(data = data,
                    property_names = ['feature_id', 'image_name'],
                    output_file = target_filename)

# define, train and deploy the classifier
c = PolygonClassifier(algorithm_params)
print 'Train classifier'
c.train(train_filename)
print 'Classify'
labels, scores = c.classify(target_filename)

# write results to geojson
out_filename = '_'.join([catalog_id, 'classified.geojson'])
jt.write_properties_to_geojson(data = zip(labels, scores), 
                               property_names = ['class_name', 'score'], 
                               input_file = target_filename,
                               output_file = out_filename)
