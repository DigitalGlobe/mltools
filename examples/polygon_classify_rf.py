# Train and deploy a polygon classifier and write results to geojson.
# Typical classification example: polygon 'includes pool' or 'does not include pool'
# Classification is performed by computing a feature vector per polygon and 
# passing feature vector to a Random Forest Classifier. 
# Polygons can be of arbitrary shape.


import numpy as np

from mltools import features
from mltools import geojson_tools as gt
from mltools import data_extractors as de

from sklearn.ensemble import RandomForestClassifier

# suppress annoying warnings
import warnings
warnings.filterwarnings('ignore')

train_file = 'train_sample.geojson'
target_file = 'target_sample.geojson'
out_file = 'classified_sample.geojson'

# Create classifier object.
# n_estimators is the number of trees in the random forest.
c = RandomForestClassifier(n_estimators = 100)

# You can create your own compute_features function here
# or use one of the available functions in mltools.features
compute_features = features.pool_basic

# get train and target data
# the point of returning target ids is that valid data can not be extracted
# for all geometries in the target file
print 'Read data'
train_rasters, train_ids, train_labels = de.get_data(train_file, return_labels=True, mask=True)
target_rasters, target_ids = de.get_data(target_file)

print 'Compute features'
feature_vectors = []
for raster in train_rasters+target_rasters:
    feature_vectors.append(compute_features(raster))

X = feature_vectors[:len(train_rasters)]
Z = feature_vectors[len(train_rasters):]

print 'Train classifier'
X, y = np.nan_to_num(np.array(X)), np.array(train_labels)    # make sure to replace NaN with finite numbers
c.fit(X, y)
class_names = c.classes_
print class_names

print 'Deploy classifier' 
Z = np.nan_to_num(np.array(Z))                               # make sure to replace NaN with finite numbers
distributions = c.predict_proba(Z)
inds = np.argmax(distributions, 1)
predicted_labels, scores = class_names[inds], distributions[range(len(inds)),inds] 

# Write results to geojson
gt.write_properties_to(data = zip(predicted_labels, scores), 
                       property_names = ['class_name', 'score'], 
                       input_file = target_file,
                       output_file = out_file,
                       filter={'feature_id':target_ids})
