# Deploy a CNN-based chip detector.
# The detector uses a sliding window and a trained chip classifier. 
# In this example, we detect boats on a pansharpened image of Hong Kong harbor.
# Implementation is with keras.

import geoio
import json
import numpy as np

from mltools import geojson_tools as gt
from keras.models import model_from_json
from keras.optimizers import SGD
from shapely.geometry import Point

print 'Load model' 
model_filename = 'boat_water_classifier.json'
weights_filename = 'boat_water_classifier.hdf5'
with open(model_filename) as f:
    m = f.next()
    model = model_from_json(json.loads(m))
    model.load_weights(weights_filename)    

print 'Compile model'
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# get chip size
chip_size = model.get_config()[0]['config']['batch_input_shape'][2:]
chip_area = chip_size[0]*chip_size[1]

# this is a pansharpened (PS) image masked with a water mask as follows:
# 1. run protogenv2RAW on gbdx to obtain a water_mask.tif 
#    (source image HAS to be multispectral acomped)
# 2. make sure watermask has same dimension as PS:
#    gdal_translate -outsize sizex sizey water_mask.tif water_mask_resampled.tif
# 3. apply mask: de.apply_mask('1030010038CD4D00.tif', 'water_mask_resampled.tif' 
#                              '1030010038CD4D00.tif') 
image = '1030010038CD4D00.tif'
img = geoio.GeoImage(image)

# deploy model on chip batches
chip_batch, location_batch, detections = [], [], []
batch_size = 32 
stride = chip_size
chips_in_image = img.meta.pixels/chip_area

for i, (chip, location_dict) in enumerate(img.iter_window(win_size=chip_size, 
                                                          stride=stride,
                                                          return_location=True)):
    if (i+1)%1000 == 0:
        print '{}/{} chips'.format(i+1, chips_in_image)

    # if mostly land, move on
    if np.sum(chip == 0) >= chip_area/2:
        continue

    chip_batch.append(chip.astype('float32')/255)
    upper_left = location_dict['upper_left_pixel']
    location_batch.append(location_dict['upper_left_pixel'])
    if len(chip_batch) == batch_size:
        chip_batch, location_batch = np.array(chip_batch), np.array(location_batch)
        prob_dist_batch = model.predict(chip_batch)
        detection_batch = location_batch[prob_dist_batch[:, 0] > prob_dist_batch[:, 1]]
        detections.extend(detection_batch)
        chip_batch, location_batch = [], []

# convert center pixel coordinates to hex-encoded (lng,lat) and write to geojson
no_detected = len(detections)
if no_detected > 0:
    results = [Point(img.raster_to_proj(*[d[0]+chip_size[0]/2,
                                          d[1]+chip_size[1]/2])).wkb.encode('hex') for d in detections]
    data = zip(results, range(no_detected), ['boat']*no_detected)
    gt.write_to(data=data, property_names=['feature_id','class_name'], 
                           output_file='detected_boats.geojson') 
else:
    print 'There are no boat detections!'
    sys.exit(0)
