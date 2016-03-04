'''
MLA: Polygon classifier for pools. 
     Two classes: (1) Polygon with swimming pool. 
                  (2) Polygon without swimming pool.
Author: Kostas Stamatiou
Created: 02/16/2016
Contact: kostas.stamatiou@digitalglobe.com
'''

import sys
import os
import numpy as np
import json 

import feature_extractors as fe
import pixel_extractors as pe

from sklearn.ensemble import RandomForestClassifier    

import math 


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
        # if there is something weird, pass
        if math.isnan(np.linalg.norm(feature_vector)): continue        
        features.append(feature_vector)
        labels.append(label)

    # train classifier
    X, y = np.array(features), np.array(labels)
    classifier.fit(X, y)

    print 'Done!'    
    return classifier


def compute_mean_accuracy(polygon_file, raster_file, classifier):
    """Deploy classifier and compute mean classification accuracy
       across classes. polygon_file must contain labels.

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
    labels, error_counter = [], 0
    for (feat, poly, data, tentative_label) in pe.extract_data(polygon_file, 
                                                               raster_file):        
        feature_vector = fe.pool_features(data, raster_file)
        try:
            # classifier prediction looks like array([]), 
            # so we need the first entry: hence the [0] 
            label = classifier.predict(feature_vector)[0]    
        except ValueError:
            label = ''                       
        labels.append(label)

        if label != tentative_label:
            error_counter += 1
            
    accuracy = float(len(labels) - error_counter)/len(labels)        
    
    print 'Done!'    
    return accuracy


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

    # get number of features to be classified
    shp = ogr.Open(polygon_file)
    lyr = shp.GetLayer()
    no_features = lyr.GetFeatureCount()

    # compute feature vectors for each polygon
    labels, counter = [], 0
    for (feat, poly, data, tentative_label) in pe.extract_data(polygon_file, 
                                                               raster_file):        
        feature_vector = fe.pool_features(data, raster_file)
        try:
            # classifier prediction looks like array([]), 
            # so we need the first entry: hence the [0] 
            label = classifier.predict(feature_vector)[0]    
        except ValueError:
            label = ''                       
        labels.append(label)

        counter += 1

        if counter % 1000 == 0:
            print counter
            

    print 'Done!'    
    return labels  


def compute_tomnod_priority(label, score):
    """Compute a priority value to be used on tomnod if a feature classified
       by the machine is to be inspected by the crowd. This is a custom 
       function and should be defined based on use case. 
       Priority is a non-negative number.
       Features on Tomnod are ordered in ascending priority order 
       (i.e. priority 0 means highest priority). 

       Args:
           label (str): The feature label.
           score (float): Confidence score from 0 to 1.

       Returns:
           Priority value (float).
    """

    # we want to prioritize polygons that the machine thinks have
    # swimming pools; this will help weed out false positives
    # is the machine thinks there are no swimming pools, we prioritize
    # by score
    if label == 'Swimming pool':
        priority = 0.0
    else:
        priority = abs(score - 0.5)

    return priority     


def classify_w_scores(polygon_file, raster_file, classifier):
    """Deploy classifier and output estimated labels with confidence scores.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           classifier (object): Instance of one of many supervised classifier
                                classes supported by scikit-learn.
       
       Returns:
           Labels (list), scores (numpy vector) and priorities (numpy vector).   
    """

    class_names = classifier.classes_
    print class_names

    # go through each polygon, compute features and classify
    labels, scores, priorities = [], [], []
    for (feat, poly, data, tentative_label) in pe.extract_data(polygon_file, 
                                                               raster_file):        
        feature_vector = fe.pool_features(data, raster_file)
        try:
            # classifier prediction looks like [[]], 
            # so we need the first entry: hence the [0]    
            probs = classifier.predict_proba(feature_vector)[0] 
            ind = np.argmax(probs)    
            label, score = class_names[ind], probs[ind]
            priority = compute_tomnod_priority(label, score)
        except ValueError:
            label, score, priority = '', 1.0, 0                               
        labels.append(label)
        scores.append(score)
        priorities.append(priority)
       
    print 'Done!'    
    return labels, np.array(scores), np.array(priorities)


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
    labels, scores, priorities = classify_w_scores(target_file, image_file, 
                                                   trained_classifier)
    
    print "Write results"    
    values = zip(labels, scores, priorities)
    jt.write_values_to_geojson(values, 
                               ['class_name', 'score', 'tomnod_priority'], 
                               target_file, 
                               output_file)

    # Compute confusion matrix; this makes sense only if the target file
    # contains known labels
    print "Confusion matrix"
    C = jt.confusion_matrix_two_geojsons(target_file, output_file)
    print C

    print "Normalized confusion matrix"
    print C.astype(float)/C.sum(1)[:, None]

    print "Done!"
