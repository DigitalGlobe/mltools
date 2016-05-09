import json
import numpy as np

from mltools import features
from mltools import geojson_tools as gt
from mltools.polygon_classifier import PolygonClassifier

# suppress annoying warnings
import warnings
warnings.filterwarnings('ignore')

train_file = 'train_small.geojson'
test_file = 'test_small.geojson'
target_file = 'target_small.geojson'

# instantiate polygon classifier
# n_estimators is the number of trees in the random forest
c = PolygonClassifier(n_estimators = 100)

# override default feature extraction method
# you can create your own feature extraction function here
# and override the default class method
c.compute_features = features.pool_basic

print 'Train classifier'
c.train(train_file, verbose=True)

print 'Test classifier'
test_labels, predicted_labels, C = c.test(test_file, verbose=True)

print 'Confusion matrix:', C

print 'Classify unknown polygons'
labels, scores = c.deploy(target_file, verbose=True)

# write results to geojson
out_file = 'classified_small.geojson'
gt.write_properties_to(data = zip(labels, scores), 
                       property_names = ['class_name', 'score'], 
                       input_file = target_file,
                       output_file = out_file)
