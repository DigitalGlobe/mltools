# Train and test a CNN-based chip classifier. 
# In this example, we classify chips from 
# a pansharpened image of Hong Kong harbor.
# Implementation is with keras.

# suppress annoying warnings
import warnings
warnings.filterwarnings('ignore')

import geoio
import json
import numpy as np
import sys

from mltools import features
from mltools import geojson_tools as gt
from mltools import data_extractors as de

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from shapely.geometry import Point

train_file = 'boats.geojson'
# this is a pansharpened (PS) image masked with a water mask as follows:
# 1. run protogenv2RAW on gbdx to obtain a water_mask.tif 
#    (source image HAS to be multispectral acomped)
# 2. make sure watermask has same dimension as PS:
#    gdal_translate -outsize sizex sizey water_mask.tif water_mask_resampled.tif
# 3. apply mask: de.apply_mask('1030010038CD4D00.tif', 'water_mask_resampled.tif' 
#                              '1030010038CD4D00.tif') 
image = '1030010038CD4D00.tif'

# specify CNN parameters
batch_size = 32
nb_classes = 2
nb_epoch = 10
img_channels = 3                          # RGB
chip_size = [60, 60]                      # chip size
half_chip_size = [x/2 for x in chip_size]

# construct CNN
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, chip_size[0], chip_size[1])))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# get boat train data from a geojson with point coordinates by extracting
# an image chip centered at each point
print 'Collect boat chips'
boat_chips, _, _ = de.get_data(train_file, return_labels=True, 
                               buffer=half_chip_size)
# note that the returned chip has dimension chip_size+1; we fix that
boat_chips = [x[:,:-1,:-1] for x in boat_chips] 
no_boats = len(boat_chips)

# split in train and test
train_total_ratio = 0.8
no_train = int(no_boats * train_total_ratio)
train_boat_chips, test_boat_chips = boat_chips[:no_train], boat_chips[no_train:]

# collect random background chips --- this is the 'noise' class
# we only keep a chip if it mostly includes water
print 'Collect background chips'
img = geoio.GeoImage(image)
no_noise = no_boats           # background chips = boat chips
noise_chips, locations = [], []
# invoke iter_window_random iterator
for chip, location_dict in img.iter_window_random(win_size=chip_size, 
                                                  no_chips=no_noise,
                                                  return_location=True):
    noise_chips.append(chip)
    upper_left = location_dict['upper_left_pixel']
    center = [x+y for x,y in zip(upper_left, half_chip_size)]
    location = Point(img.raster_to_proj(*center))      # location in (lng, lat)  
    locations.append(location.wkb.encode('hex'))       # encode in hex

# create a geojson for visualization purposes
data = zip(locations, range(len(locations)), ['noise']*no_noise)
gt.write_to(data=data, property_names=['feature_id','class_name'], 
                       output_file='noise.geojson') 

# split in train and test
no_train = int(no_noise * train_total_ratio)
train_noise_chips, test_noise_chips = noise_chips[:no_train], noise_chips[no_train:]

# prepare arrays
X_train = np.array(list(train_boat_chips) + list(train_noise_chips)).astype('float32')/255
X_test = np.array(list(test_boat_chips) + list(test_noise_chips)).astype('float32')/255
# generate label arrays in categorical format
y_train, y_test = np.zeros((len(X_train),2), dtype=int), np.zeros((len(X_test),2), dtype=int)
y_train[:len(train_boat_chips),0] = 1
y_train[len(train_boat_chips):,1] = 1
y_test[:len(test_boat_chips),0] = 1
y_test[len(test_boat_chips):,1] = 1

print 'Train model'
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          validation_data=(X_test, y_test),
          verbose=1)
    
print 'Test model'
score = model.evaluate(X_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# save model for future use
name = 'boat_water_classifier'
model_filename = '{}.json'.format(name)
weight_filename = '{}.hdf5'.format(name)
json_string = model.to_json()
with open(model_filename, 'w') as f:
    json.dump(json_string, f)
weight_filename = '{}.hdf5'.format(name)
model.save_weights(weight_filename)
