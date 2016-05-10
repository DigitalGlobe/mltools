# Pull train and target data from Tomnod and store in geojson format.

import json

from mltools.crowdsourcing import TomnodCommunicator
from mltools import geojson_tools as gt

# Initialize Tomnod communicator class --- you need credentials for this.
tc = TomnodCommunicator(json.load(open('credentials.json')))

# Specify Tomnod schema. This schema corresponds to a Tomnod campaign to 
# classify polygons as containing a swimming pool or not. 
schema = 'adelaide_pools_2016'

# specify classes
class_names = ['Swimming pool', 'No swimming pool']

# Fetch training data from each class. 
# We request polygons from each class classified by the crowd 
# with high confidence score.
print 'Collect training data'  
train_filenames = ['train_pool.geojson', 'train_no_pool.geojson']
for i, class_name in enumerate(class_names):
    data = tc.get_classified(class_name, 
                             schema, 
                             score_range=[0.95,1.00], 
                             max_number=1000)
    gt.write_to(data, 
                property_names=['feature_id', 'image_id', 'class_name'],
                output_file=train_filenames[i])

# make one training file
gt.join(train_filenames, 'train.geojson')

# Fetch target samples. 
# We request polygons which have not been classified by the crowd.
print 'Collect target data'
data = tc.get_unclassified(schema)

# make target file
gt.write_to(data, 
            property_names=['feature_id', 'image_id'],
            output_file='target.geojson')