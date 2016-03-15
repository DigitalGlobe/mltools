# Script that uses the PolygonClassifier class to find all polygons from 
# a given image that contain swimming pools.
# The script pulls a set of classified polygons from Tomnod in order to train 
# the classifier and then deploys the classifier on a set of unlabeled polygons.
# The output of the script is a geojson file with the newly labeled polygons. 

# This script was used in the adelaide_pools_2016 campaign. 

import geojson
import json
import os
import sys

import crowdsourcing as cr
import json_tools as jt

from polygon_classifier import PolygonClassifier
from crowdsourcing import TomnodCommunicator

# suppress annoying warnings
import warnings
warnings.filterwarnings("ignore")

# tomnod credentials


# get job parameters from json
with open('job.json', 'r') as f:
    job = json.load(f)

schema = job['schema']
cat_id = job['cat_id']
no_train_pools = job['no_train_pools']
no_train_nopools = job['no_train_nopools']
max_polys_to_classify = job['max_polygons'] 
max_poly_area = job['max_area']             # don't consider polys too large 
algorithm_params = job['algorithm_params'] 

image_file = cat_id + '.tif'
train_file = cat_id + '_train.geojson'
test_file = cat_id + '_test.geojson'
target_file = cat_id + '_target.geojson'
output_file = cat_id + '.geojson'

print 'Get ground truth'

with open('credentials.json', 'r') as f:
    credentials = json.load(f)


cr.train_geojson(schema, 
                 cat_id, 
                 no_train_pools, 
	               'gt_pools.geojson', 
                 'Swimming pool', 
                 credentials,
                 min_votes = 1)


cr.train_geojson(schema, 
                 cat_id, 
                 no_train_nopools, 
	               'gt_nopools.geojson', 
                 'No swimming pool', 
                 credentials, 
                 max_area=max_area,
                 min_votes = 1)

jt.join_two_geojsons('gt_pools.geojson', 
                     'gt_nopools.geojson', 
                     train_file)

# get target file  
cr.target_geojson(schema, 
                  cat_id, 
                  target_file, 
                  credentials,
                  max_number = max_polygons, 
                  max_votes = 0,
                  max_area = max_area 
                  )


# run pool detector
print 'Run detector'
pcp.main('job.json
