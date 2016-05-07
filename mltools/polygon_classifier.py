# Polygon classifier. Classifies a set of polygons on an image.

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
            classifier_name = parameters['classifier']
        except KeyError:
            classifier_name = 'random forest' 
        
        if classifier_name.lower() ==  'random forest':
            self.classifier = sklearn.ensemble.RandomForestClassifier()
            try:
                self.classifier.n_estimators = parameters['no_trees']
            except KeyError:         
                pass


    def feature_extractor(self, data):
        '''The simplest feature extractor. This should be overriden 
           based on use case when the class is instantiated.
           Args:
               data (numpy array): Pixel data vector.
           Returns:
               A vector with the mean, std and variance of data.
        '''

        return [np.mean(data), np.std(data), np.var(data)]
            
                

    def train(self, train_file, classifier_pickle_file = ''):
        '''Train classifier.
           Args:
               train_file (str): Training data filename (geojson).
               classifier_pickle_file (str): File to store classifier pickle.
                                             If empty, don't store.   
        '''
       
        # compute feature vector for each polygon
        features, labels = [], []
        for pixels, label in extract_data(polygon_file = train_file):        
            # if there is no data or label, skip since training is impossible
            if pixels is None or label is None: continue    
            feature_vector = self.feature_extractor(pixels)
            # if for any reason the norm of the feature vector is not defined, skip
            if math.isnan(np.linalg.norm(feature_vector)): continue        
            
            features.append(feature_vector)
            labels.append(label)

        X, y = np.array(features), np.array(labels)

        # train classifier
        self.classifier.fit(X, y)
        
        if classifier_pickle_file:
            with open(classifier_file, 'w') as f:
                pickle.dump(classifier, f)
              

    def test(self, test_file):
        '''Deploy classifier on a test file and output confusion matrix. 
           Args:
               test_file (str): Test filename (geojson).
           Returns:
               List of test labels, list of predicted labels and numpy
               confusion matrix.   
        '''

        class_names = self.classifier.classes_
        test_labels, predicted_labels = [], []

        # for each polygon, compute feature vector and classify
        for pixels, test_label in extract_data(polygon_file = test_file):
            if pixels is None or test_label not in class_names:
                continue       
            else:
                feature_vector = self.feature_extractor(pixels)
                # classifier prediction looks like array([]), 
                # so we need the first entry: hence the [0] 
                try:
                    prob_distr = self.classifier.predict_proba(feature_vector)[0] 
                    ind = np.argmax(prob_distr)    
                    predicted_label, score = class_names[ind], prob_distr[ind]                
                except ValueError:
                    continue
            test_labels.append(test_label)
            predicted_labels.append(predicted_label)
        
        C = confusion_matrix(test_labels, predicted_labels)
        return test_labels, predicted_labels, C


    def deploy(self, target_file):
        '''Deploy classifier on target_file with unknown labels. 
           Args:
               target_file (str): Target filename (geojson).
           Returns:
               List of predicted labels and list of scores.   
        '''

        class_names = self.classifier.classes_
        predicted_labels, scores = [], [] 
        
        # for each polygon, compute feature vector and classify
        # note that label is empty in this case
        for pixels, label in extract_data(polygon_file=target_file):       
            
            if data is None:
                predicted_label, score = None, None
            else:
                feature_vector = self.feature_extractor(data)
                # classifier prediction looks like array([]), 
                # so we need the first entry: hence the [0] 
                try:
                    prob_distr = self.classifier.predict_proba(feature_vector)[0] 
                    ind = np.argmax(prob_distr)    
                    predicted_label, score = class_names[ind], prob_distr[ind]                
                except ValueError:
                    # if classification went wrong
                    predicted_label, score = None, None		
        
            predicted_labels.append(predicted_label)
            scores.append(score)
            
        return predicted_labels, scores
        
