# Train, test, deploy polygon classifier and write results to geojson

from mltools import features
from mltools import geojson_tools as gt
from mltools.polygon_classifier import PolygonClassifier
# suppress annoying warnings
import warnings
warnings.filterwarnings('ignore')

train_file = 'train.geojson'
test_file = 'test.geojson'
target_file = 'target.geojson'

# Create PolygonClassifier object.
# n_estimators is the number of trees in the random forest.
c = PolygonClassifier(n_estimators = 100)

# Override default PolygonClassifier compute_features.
# You can create your own compute_features here
# or use one of the available functions in mltools.features
c.compute_features = features.pool_basic

print 'Train classifier'
c.train(train_file, verbose=True)

print 'Test classifier'
test_labels, predicted_labels, C = c.test(test_file, verbose=True)

print 'Confusion matrix:', C

print 'Classify unknown polygons'
labels, scores = c.deploy(target_file, verbose=True)

# Write results to geojson.
out_file = 'classified.geojson'
gt.write_properties_to(data = zip(labels, scores), 
                       property_names = ['class_name', 'score'], 
                       input_file = target_file,
                       output_file = out_file)