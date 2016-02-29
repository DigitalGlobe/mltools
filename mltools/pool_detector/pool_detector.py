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
        feature_vector = fe.pool_features(data, raster_file)
        features.append(feature_vector)
        labels.append(label)
                        
    # train classifier
    X, y = np.array(features), np.array(labels)
    classifier.fit( X, y )

    print 'Done!'    
    return classifier


def classify(polygon_file, raster_file, classifier):
    """Deploy classifier and output estimated labels.

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
        feature_vector = fe.pool_features(data, raster_file)
        try:
            label = classifier.predict(feature_vector)    
        except ValueError:
            label = ''                       
        labels.append(label)

    print 'Done!'    
    return labels  


def classify_w_scores(polygon_file, raster_file, classifier):
    """Deploy classifier and output estimated labels with confidence scores.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           classifier (object): Instance of one of many supervised classifier
                                classes supported by scikit-learn.
       
       Returns:
           List of labels (list) and vector of scores (numpy vector).                                             
    """

    class_names = classifier.classes_

    # compute feature vectors for each polygon
    labels, scores = [], []
    for (feat, poly, data, label) in pe.extract_data(polygon_file, raster_file):        
        feature_vector = fe.pool_features(data, raster_file)
        try:
            probs = classifier.predict_proba(feature_vector) 
            ind = np.argmax(probs)
            label, score = class_names[ind], probs[ind]
        except ValueError:
            label, score = '', 1.0                       
        labels.append(label)
        scores.append(score)

    print 'Done!'    
    return labels, np.array(scores)  


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
    n_estimators = algo_params["n_estimators"]
    oob_score = algo_params["oob_score"]
    class_weight = algo_params["class_weight"]
    classifier = RandomForestClassifier(n_estimators = n_estimators, 
                                        oob_score = oob_score, 
                                        class_weight = class_weight)
        
    print "Train model"
    trained_classifier = train_model(train_file, image_file, classifier)
    
    print "Classify"
    # labels = classify(target_file, image_file, trained_classifier)
    labels, scores = classify_w_scores(target_file, image_file, 
                                       trained_classifier)

    print "Write results"    
    jt.write_values_to_geojson(labels, 'class_name', target_file, output_file)
    jt.write_values_to_geojson(scores, 'score', target_file, output_file)

    # Compute confusion matrix; this makes sense only if the target file
    # contains known labels
    print "Confusion matrix"
    C = jt.confusion_matrix_two_geojsons(target_file, output_file)
    print C

    print "Normalized confusion matrix"
    print C.astype(float)/C.sum(1)[:, None]

    print "Done!"