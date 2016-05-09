# Classifies a set of polygons on an image.
# Inherits from scikit-learn Random Forest Classifier. 

import math 
import numpy as np
import pickle

from data_extractors import extract
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
   

class PolygonClassifier(RandomForestClassifier):


    def compute_features(self, data):
        '''Computes a simple feature vector. This should be overriden 
           based on use case when the class is instantiated.
           Args:
               data (numpy array): Pixel data vector.
           Returns:
               A vector with the mean, std and variance of data.
        '''

        return [np.mean(data), np.std(data), np.var(data)]
            
                
    def train(self, train_file, classifier_pickle_file='', verbose=False):
        '''Train classifier.
           Args:
               train_file (str): Training data filename in mltools geojson format.
               classifier_pickle_file (str): File to store classifier pickle.
                                             If empty, don't store.
               verbose (bool): Print error messages and counter to stdout.   
        '''
       
        if verbose:
            def verboseprint(arg):
                print arg
        else:
            verboseprint = lambda a: None       
        
        feature_vectors, labels, counter = [], [], 0
        for pixels, label in extract(polygon_file=train_file, return_class=True):        
            try:     
                feature_vector = self.compute_features(pixels)
            # if the feature vector can not be computed, skip
            except ValueError, e:
                verboseprint('{}: Cannot compute features!'.format(e))
                continue           
            # if the norm of the feature vector is not defined, skip
            if math.isnan(np.linalg.norm(feature_vector)):
                verboseprint('Norm of feature vector not defined!') 
                continue        
            
            feature_vectors.append(feature_vector)
            labels.append(label)
            counter += 1
            if counter%5000==0: verboseprint(counter)
  
        X, y = np.array(feature_vectors), np.array(labels)
        # train classifier
        self.fit(X, y)
        
        if classifier_pickle_file:
            with open(classifier_file, 'w') as f:
                pickle.dump(classifier, f)
              

    def test(self, test_file, verbose=False):
        '''Deploy classifier on a test file and output confusion matrix. 
           Args:
               test_file (str): Test data filename in mltools geojson format.
               verbose (bool): Print error messages and counter to stdout.
           Returns:
               List of test labels, list of predicted labels and numpy
               confusion matrix.   
        '''

        if verbose:
            def verboseprint(arg):
                print arg
        else:
            verboseprint = lambda a: None

        class_names = self.classes_
        test_labels, predicted_labels = [], []
        counter = 0

        # for each polygon, compute feature vector and classify
        for pixels, test_label in extract(polygon_file=test_file,
                                          return_class=True):
            try:
                feature_vector = self.compute_features(pixels) 
            # if the feature vector can not be computed, skip
            except ValueError, e:
                verboseprint('{}: Cannot compute features!'.format(e))
                continue                
            # classifier prediction looks like array([]), 
            # so we need the first entry: hence the [0] 
            try:
                prob_distr = self.predict_proba(feature_vector)[0] 
                ind = np.argmax(prob_distr)    
                predicted_label, score = class_names[ind], prob_distr[ind]                
            except ValueError, e:
                verboseprint('{}: Cannot classify!'.format(e))
                continue
            test_labels.append(test_label)
            predicted_labels.append(predicted_label)
            counter += 1
            if counter%1000==0: verboseprint(counter)   
        
        C = confusion_matrix(test_labels, predicted_labels)

        return test_labels, predicted_labels, C


    def deploy(self, target_file, verbose=False):
        '''Deploy classifier on target_file. 
           Args:
               target_file (str): Target data filename in mltools geojson format.
               verbose (bool): Print error messages and counter to stdout.
           Returns:
               List of predicted labels and list of scores.   
        '''

        if verbose:
            def verboseprint(arg):
                print arg
        else:
            verboseprint = lambda a: None

        class_names = self.classes_
        predicted_labels, scores, counter = [], [], 0 
        
        # for each polygon, compute feature vector and classify
        # note that label is empty in this case
        for pixels in extract(polygon_file=target_file):       
            try:
                feature_vector = self.compute_features(pixels)
            # if the feature vector can not be computed, skip
            except ValueError, e:
                verboseprint('{}: Cannot compute features!'.format(e))
                continue
            # classifier prediction looks like array([]), 
            # so we need the first entry: hence the [0] 
            try:
                prob_distr = self.predict_proba(feature_vector)[0] 
                ind = np.argmax(prob_distr)    
                predicted_label, score = class_names[ind], prob_distr[ind]                
            except ValueError, e:
                # if classification went wrong
                verboseprint('{}: Cannot classify!'.format(e)) 
                predicted_label, score = None, None		
            predicted_labels.append(predicted_label)
            scores.append(score)
            counter += 1
            if counter%1000==0: verboseprint(counter)
             
        return predicted_labels, scores
       
