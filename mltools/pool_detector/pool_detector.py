'''
Pool detector built after simple_classifier.
'''

import sys
import os
import numpy as np
import json 
import geojson

from .. import json_tools as jt
from .. import feature_extractors as fe
from .. import pixel_extractors as pe

from sklearn.ensemble import RandomForestClassifier     

def train_model(polygon_file, raster_file, classifier):
    """Train classifier and output classifier parameters.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           classifier (object): Instance of one of many supervised classifier
                                classes supported by scikit-learn.
       
       Returns:
           Trained classifier (object).                                             
    """

    # get water signature

    # compute feature vectors for each polygon
    features = []
    labels = []
    for (feat, poly, data, label) in pe.extract_data(polygon_file, raster_file):        
        for featureVector in fe.pool_features(data, raster_file):
            features.append(featureVector)
            labels.append(label)
            print label, featureVector
            
    # train classifier
    X, y = np.array(features), np.array(labels)
    # train
    classifier.fit( X, y )
    # # store model
    # with open(classifier_file,"w") as fh:
    #     pickle.dump( classifier, fh )
    print 'Done!'    
    return classifier


def classify(polygon_file, raster_file, classifier):
    """Deploy classifier and output corresponding list of labels.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           classifier (object): Instance of one of many supervised classifier
                                classes supported by scikit-learn.
       
       Returns:
           List of labels (list).                                             
    """

    # compute feature vectors for each polygon
    labels = []
    for (feat, poly, data, label) in pe.extract_data(polygon_file, raster_file):        
        for featureVector in fe.pool_features(data, raster_file):
            labels_this_feature = classifier.predict(featureVector)                        
        labels.append(labels_this_feature[0])

    print 'Done!'    
    return labels     

  
def main(job_file):
    """Runs the simple_lulc workflow.

       Args:
           job_file (str): Job filename (.json, see README of this repo) 
    """    
   
    # get job parameters
    job = json.load(open(job_file, 'r'))
    image_file = job["image_file"]
    train_file = job["train_file"]
    target_file = job["target_file"]
    output_file = job["output_file"]
    algo_params = job["params"]       # these are parameters pertinent to the 
                                      # algorithm
    
    # Using random forest classifier
    no_trees = algo_params["no_trees"]
    classifier = RandomForestClassifier(n_estimators = no_trees)
        
    print "Train model"
    trained_classifier = train_model(train_file, 
                                     image_file, 
                                     classifier)
    
    print "Classify"
    labels = classify(target_file, 
                      image_file, 
                      trained_classifier)
                                        
    print "Write results"    
    jt.write_labels_to_geojson(labels, target_file, output_file)

    print "Confusion matrix"
    C = jt.confusion_matrix_two_geojsons(target_file, output_file)

    print C

    print "Done!"