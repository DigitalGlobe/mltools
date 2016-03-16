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
max_area = job['max_area']                            # max polygon area in m2
max_polys_to_classify = job['max_polys_to_classify']
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
    no_samples = class_entry['no_samples']    
    print 'Collect', str(no_samples), 'samples for class', class_name, 
    'from', schema, 'and image', catalog_id 

    train_filenames.append('_'.join([class_name, catalog_id, 'train.geojson']))
    data = tc.get_high_confidence_features(campaign_schema = schema, 
                                           image_id = catalog_id, 
                                           class_name = class_name,
                                           max_number = no_samples,
                                           write_to = train_filenames[i])
    
# assemble final train file by joining constituent train files
train_filename = '_'.join([catalog_id, 'train.geojson'])
jt.join_geojsons(train_filenames, train_filename)

# fetch unclassified features for classification
target_filename = '_'.join([catalog_id, 'target.geojson'])
data = tc.get_low_confidence_features(campaign_schema = schema, 
                                      image_id = catalog_id, 
                                      max_number = no_samples,
                                      write_to = target_filename)

# define, train and deploy the classifier
c = PolygonClassifier(algorithm_params)
c.train(train_filename)
labels, scores = c.classify(target_filename)

# write results to geojson
out_filename = '_'.join([catalog_id, 'classified.geojson'])
jt.write_properties_to_geojson(data = zip(labels, scores), 
                               property_names = ['class_name', 'score'], 
                               input_file = target_filename,
                               output_file = out_filename)