# A little python script that uses mltools to train a pool detector and classify
# polygons in a given overlay

import json
import os

import pool_detector as pd
import crowdsourcing as cr
import json_tools as jt

# suppress annoying warnings
import warnings
warnings.filterwarnings("ignore")

# parameters
credentials = {'host':'mapperdb.cj6xoak5f54o.us-east-1.rds.amazonaws.com',
               'db':'tomnod', 
               'user':'tomnod', 
               'password':'5cqJUfOXmEufeexwJ2EW4XCE'}

schema = 'adelaide_pools_2016'
cat_id = '1040010015D38800'
image_file = '14316.tif'
max_pools_samples = 300
ratio_train_pools = 0.333
max_nopools_samples = 5000
ratio_train_nopools = 0.02
train_file = 'train.geojson'
target_file = 'target.geojson'
output_file = 'classified.geojson'
max_area = 1000                           # max area in m2

# classifier params
n_estimators = 200
oob_score = False
class_weight = None

# get ground truth for pools and no pools 
print 'Get GT'
cr.train_geojson(schema, cat_id, max_pools_samples, 
	               'gt_pools.geojson', 'Swimming pool', credentials)

cr.train_geojson(schema, cat_id, max_nopools_samples, 
	               'gt_nopools.geojson', 'No swimming pool', 
                 credentials, max_area=max_area)

# separate in train and test
jt.split_geojson('gt_pools.geojson', 'train_pools.geojson', 
	             'test_pools.geojson', ratio_train_pools)
jt.split_geojson('gt_nopools.geojson', 'train_nopools.geojson', 
	             'test_nopools.geojson', ratio_train_nopools)
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
pd.main('job.json')