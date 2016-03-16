# Script that uses the PolygonClassifier class to find all polygons from 
# a given image that contain swimming pools.
# The script pulls a set of classified polygons from Tomnod in order to train 
# the classifier and then deploys the classifier on a set of unclassified polygons.
# The output of the script is a geojson file with the newly classified polygons. 
# This script was used in the adelaide_pools_2016 campaign. 

import json
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
cat_id = job['cat_id']
no_train_pools = job['no_train_pools']
no_train_nopools = job['no_train_nopools']
max_area = job['max_area']           # max area in m2
max_polygons = job['max_polygons']   # max polygons to be classified
algorithm_params = job['algorithm_params']

image_file = cat_id + '.tif'
train_file = cat_id + '_train.geojson'
target_file = cat_id + '_target.geojson'
output_file = cat_id + '.geojson'

# get ground truth for pools and no pools 
print 'Get GT'

# get tomnod credentials
with open('credentials.json', 'r') as f:
    credentials = json.load(f)

# initialize Tomnod communicator class
tc = TomnodCommunicator(credentials)

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

print 'Train and test model'
c = PolygonClassifier(algorithm_params)
c.train(train_file)


#### more stuff, potentially useful for final form of this script

# def main(job_file):
#     """Runs the simple_lulc workflow.

#        Args:
#            job_file (str): Job filename (.json, see README of this repo) 
#     """    
   
#     # get job parameters
#     job = json.load(open(job_file, 'r'))
#     image_file = job["image_file"]
#     train_file = job["train_file"]
#     target_file = job["target_file"]
#     output_file = job["output_file"]
#     algo_params = job["params"]       # these are parameters pertinent to the 
#                                       # algorithm
    
#     # Using random forest classifier
#     n_estimators = algo_params["n_estimators"]
#     oob_score = algo_params["oob_score"]
#     class_weight = algo_params["class_weight"]
#     classifier = RandomForestClassifier(n_estimators = n_estimators, 
#                                         oob_score = oob_score, 
#                                         class_weight = class_weight)
        
#     print "Train model"
#     trained_classifier = train_model(train_file, image_file, classifier)
    
#     print "Classify"
#     labels, scores, priorities = classify_w_scores(target_file, image_file, 
#                                                    trained_classifier)
    
#     print "Write results"    
#     values = zip(labels, scores, priorities)
#     jt.write_values_to_geojson(values, 
#                                ['class_name', 'score', 'tomnod_priority'], 
#                                target_file, 
#                                output_file)

#     # Compute confusion matrix; this makes sense only if the target file
#     # contains known labels
#     print "Confusion matrix"
#     C = jt.confusion_matrix_two_geojsons(target_file, output_file)
#     print C

#     print "Normalized confusion matrix"
#     print C.astype(float)/C.sum(1)[:, None]

#     print "Done!"
    


