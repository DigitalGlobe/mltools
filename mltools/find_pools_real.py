'''
What: a python script that deploys polygon_classifier_pools on an actual target
      set.
Author: Kostas Stamatiou
Created: 03/04/2016
Contact: kostas.stamatiou@digitalglobe.com
'''

import json
import os
import sys


import polygon_classifier_pools as pcp
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
no_train_pools = class_job['no_train_pools']
no_train_nopools = class_job['no_train_nopools']
train_file = cat_id + '_train_real.geojson'
target_file = cat_id + '_target_real.geojson'
output_file = cat_id + '_real.geojson'
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
jt.join_two_geojsons('gt_pools.geojson', 'gt_nopools.geojson', 
                   train_file)

# get target file 
cr.target_geojson(schema, cat_id, 1e06, 
                  target_file, 
                  credentials,
                  max_votes = 0,
                  max_area = max_area 
                  )

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
pcp.main('job.json')
