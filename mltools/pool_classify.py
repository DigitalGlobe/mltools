'''
What: a python script that deploys pool_classifier to train a pool classifier 
and classify polygons as containing pools or not in given 8-band image.
Author: Kostas Stamatiou
Created: 02/28/2016
Contact: kostas.stamatiou@digitalglobe.com
'''

import json
import os
import sys


import pool_classifier as pc
import crowdsourcing as cr
import json_tools as jt

# suppress annoying warnings
import warnings
warnings.filterwarnings("ignore")

# tomnod parameters
credentials = {'host':'mapperdb.cj6xoak5f54o.us-east-1.rds.amazonaws.com',
               'db':'tomnod', 
               'user':'tomnod', 
               'password':'5cqJUfOXmEufeexwJ2EW4XCE'}


with open('class_job.json', 'r') as f:
    class_job = json.load(f)

# get parameters from json
schema = class_job['schema']
cat_id = class_job['cat_id']
image_file = cat_id + '.tif'
no_pools = class_job['no_pools']
no_train_pools = class_job['no_train_pools']
no_nopools = class_job['no_nopools']
no_train_nopools = class_job['no_train_nopools']
train_file = class_job['train_file']
target_file = class_job['target_file']
output_file = cat_id + '.geojson'
max_area = class_job['max_area']                           # max area in m2

# classifier params
n_estimators = 200
oob_score = False
class_weight = None

# get ground truth for pools and no pools 
print 'Get GT'
cr.train_geojson(schema, cat_id, no_pools, 
	               'gt_pools.geojson', 'Swimming pool', credentials)

cr.train_geojson(schema, cat_id, no_nopools, 
	               'gt_nopools.geojson', 'No swimming pool', 
                 credentials, max_area=max_area)

# separate in train and test
jt.split_geojson('gt_pools.geojson', 'train_pools.geojson', 
	             'test_pools.geojson', no_train_pools)
jt.split_geojson('gt_nopools.geojson', 'train_nopools.geojson', 
	             'test_nopools.geojson', no_train_nopools)
jt.join_two_geojsons('train_pools.geojson', 'train_nopools.geojson', 
	                 train_file)
jt.join_two_geojsons('test_pools.geojson', 'test_nopools.geojson', 
	                 target_file)

# set job parameters
job_json = {"params":{"n_estimators": n_estimators, 
                      "oob_score": oob_score, 
                      "class_weight": class_weight},
            "image_file": image_file,
            "train_file": train_file,
            "target_file": target_file,
            "output_file": output_file
            }

with open('job.json', 'w') as f:
	json.dump(job_json, f)

# run pool detector
print 'Run detector'
pc.main('job.json')
