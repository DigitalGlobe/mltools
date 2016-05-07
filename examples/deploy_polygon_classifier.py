# Deploy PolygonClassifier on a set of unclassified polygons.
# The script pulls a set of classified polygons from Tomnod for training
# and a set of unclassified polygons for classification. 
# This script was used in the adelaide_pools_2016 campaign. 

import json
import numpy as np
import os
import sys

from mltools import features
from mltools import geojson_tools as gt
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
    print 'Collect {} {} samples from schema {} and image {}'.format(no_train_samples,
                                                                     class_name,
                                                                     schema,
                                                                     catalog_id)
    train_filenames.append('_'.join([class_name, catalog_id, 'train.geojson']))
    data = tc.get_high_confidence_features(campaign_schema = schema, 
                                           image_id = catalog_id, 
                                           class_name = class_name,
                                           max_number = no_train_samples)
    gt.write_to(data = data, 
                property_names = ['feature_id', 'image_id', 'class_name'],
                output_file = train_filenames[i])
    
# assemble final train file by joining constituent train files
train_filename = '_'.join([catalog_id, 'train.geojson'])
gt.join(train_filenames, train_filename)

# fetch unclassified features for classification
target_filename = '_'.join([catalog_id, 'target.geojson'])
no_polys_to_classify = target_params['no_polys_to_classify']
data = tc.get_low_confidence_features(campaign_schema = schema, 
                                      image_id = catalog_id, 
                                      max_number = no_polys_to_classify)
gt.write_to(data = data,
            property_names = ['feature_id', 'image_id'],
            output_file = target_filename)

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

print 'Classify unknown polygons'
labels, scores = c.deploy(target_filename)

# write results to geojson
out_filename = '_'.join([catalog_id, 'classified.geojson'])
gt.write_properties_to(data = zip(labels, scores), 
                       property_names = ['class_name', 'score'], 
                       input_file = target_filename,
                       output_file = out_filename)
