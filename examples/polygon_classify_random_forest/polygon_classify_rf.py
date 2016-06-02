# Train, test and deploy a polygon classifier using multispectral imagery 
# and a random forest classifier. 
# The classifier classifies polygons as 'Contains swimming pool' and 'Does not contain swimming pool'.
# Classification is performed by computing a feature vector per polygon and 
# passing feature vector to a Random Forest Classifier. 
# Polygons can be of arbitrary shape.

import numpy as np
import warnings
warnings.filterwarnings('ignore')   # suppress annoying warnings

from mltools import features
from mltools import geojson_tools as gt
from mltools import data_extractors as de
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# get train, test and target data
# the point of returning polygon ids for the target data is that 
# pixels can not always be extracted for all geometries in the file
# so we need to know the ids of the polygons that will be classified
print 'Read data'
train_rasters, _, train_labels = de.get_data('train.geojson', return_labels=True, mask=True)
test_rasters, _, test_labels = de.get_data('test.geojson', return_labels=True, mask=True)
target_rasters, target_ids = de.get_data('target.geojson')

# You can create your own compute_features function here
# or use one of the available functions in mltools.features
compute_features = features.pool_basic

print 'Compute features'
feature_vectors = []
for raster in train_rasters + test_rasters + target_rasters:
    feature_vectors.append(compute_features(raster))

X = feature_vectors[:len(train_rasters)]
Y = feature_vectors[len(train_rasters):len(train_rasters)+len(test_rasters)]
Z = feature_vectors[len(train_rasters)+len(test_rasters):]

# Create classifier object.
# n_estimators is the number of trees in the random forest.
c = RandomForestClassifier(n_estimators = 100)

print 'Train classifier'
X, train_labels = np.nan_to_num(np.array(X)), np.array(train_labels)    # make sure to replace NaN with finite numbers
c.fit(X, train_labels)
class_names = c.classes_
print class_names

print 'Test classifier'
Y, test_labels = np.nan_to_num(np.array(Y)), np.array(test_labels)    
distributions = c.predict_proba(Y)
inds = np.argmax(distributions, 1)
predicted_labels, scores = class_names[inds], distributions[range(len(inds)),inds] 
C = confusion_matrix(test_labels, predicted_labels)
print 'Confusion matrix'
print C

print 'Deploy classifier' 
Z = np.nan_to_num(np.array(Z))                               
distributions = c.predict_proba(Z)
inds = np.argmax(distributions, 1)
predicted_labels, scores = class_names[inds], distributions[range(len(inds)),inds] 

# Write results to geojson
gt.write_properties_to(data = zip(predicted_labels, scores), 
                       property_names = ['class_name', 'score'], 
                       input_file = 'target.geojson',
                       output_file = 'classified.geojson',
                       filter={'feature_id':target_ids})
