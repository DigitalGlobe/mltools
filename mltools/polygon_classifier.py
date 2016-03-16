# Polygon classifier. Classifies a set of polygons on an image.

import feature_extractors
import geojson
import math 
import numpy as np
import os
import pickle
import sklearn.ensemble
import sys

from data_extractors import extract_data
from sklearn.metrics import confusion_matrix
   

class PolygonClassifier():

    def __init__(self, parameters):
        '''
        Args:
            parameters (dict): Dictionary with algorithm parameters.
        '''                
        
        try:
            feature_extractor_name = parameters['feature_extractor']
        except KeyError:
            feature_extractor_name = 'vanilla_features'    

        self.feature_extractor = getattr(feature_extractors, feature_extractor_name)    

        try:
            classifier_name = parameters['classifier']
        except KeyError:
            classifier_name = 'random forest' 
        
        if classifier_name.lower() ==  'random forest':
            self.classifier = sklearn.ensemble.RandomForestClassifier()
            try:
                self.classifier.n_estimators = parameters['no_trees']
            except KeyError:         
                pass


    def train(self, train_file, classifier_pickle_file = ''):
        '''Train classifier.

           Args:
               train_file (str): Training data filename (geojson).
               classifier_pickle_file (str): File to store classifier pickle.
                                             If empty, don't store.   
        '''
       
        # compute feature vector for each polygon
        features, labels = [], []
        for poly, data, label in extract_data(polygon_file = train_file):        
            feature_vector = self.feature_extractor(data)
            # print feature_vector, label
            # if there is something weird, pass
            if math.isnan(np.linalg.norm(feature_vector)): 
                continue        
            
            features.append(feature_vector)
            labels.append(label)

        X, y = np.array(features), np.array(labels)
        # train classifier
        self.classifier.fit(X, y)
        
        if classifier_pickle_file:
            with open(classifier_file, 'w') as f:
                pickle.dump(classifier, f)
              

    def classify(self, target_file, max_number = -1, return_confusion_matrix = False):
        """Deploy classifier on target_file and output estimated labels
           and corresponding confidence scores. 

           Args:
               target_file (str): Target filename (geojson).
               max_number (int): Maximum number of features to classify. Default value 
                                 of -1 classifies all features in target_file.
               return_confusion_matrix (bool): If true, a confusion matrix is returned.
                                           This makes sense only when target_file includes 
                                           known labels and can be used to estimate the 
                                           classifier accuracy.            

           Returns:
               Label list, numpy score vector and numpy confusion matrix (optional).   
        """

        class_names = self.classifier.classes_
        test_labels, predicted_labels, scores, counter = [], [], [], 0 
        
        # for each polygon, compute feature vector and classify
        for poly, data, test_label in extract_data(polygon_file = target_file):       
            
            feature_vector = self.feature_extractor(data)

            try:
                # classifier prediction looks like array([]), 
                # so we need the first entry: hence the [0] 
                probability_distr = self.classifier.predict_proba(feature_vector)[0] 
                ind = np.argmax(probability_distr)    
                predicted_label, score = class_names[ind], probability_distr[ind]                
            except ValueError:
                predicted_label, score = '', 1.0                       
       
            test_labels.append(test_label)
            predicted_labels.append(predicted_label)
            scores.append(score)

            counter += 1
            if counter % 1000 == 0: print counter, 'classified'
            if counter == max_number: break		
        
        predicted_labels, scores = np.array(predicted_labels), np.array(scores)
            
        if return_confusion_matrix:
            C = confusion_matrix(test_labels, predicted_labels)
            return predicted_labels, scores, C
        else:
            return predicted_labels, scores
        
