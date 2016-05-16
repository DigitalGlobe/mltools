# Train and test a CNN-based classifier. 
# In this example, we classify chips from 
# a pansharpened image of Hong Kong harbor.
# Implementation is with keras.

# suppress annoying warnings
import warnings
warnings.filterwarnings('ignore')

import geoio
import numpy as np

from mltools import features
from mltools import geojson_tools as gt
from mltools import data_extractors as de

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

train_file = 'boats.geojson'
image = '1030010038CD4D00.tif'

# specify CNN parameters
batch_size = 32
nb_classes = 2
nb_epoch = 1
img_rows, img_cols = 60, 60
chip_size = [img_rows, img_cols]
img_channels = 3  # RGB

# construct CNN
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
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
                               buffer=[x/2 for x in chip_size])

# split in train and test
no_train = int(len(boat_chips)*0.8)
train_boat_chips, test_boat_chips = boat_chips[:no_train], boat_chips[no_train:]

# collect random background chips --- this is the 'noise' class
print 'Collect background chips'
noise_chips = de.random_window(image, chip_size=chip_size, 
                                      no_chips=len(boat_chips))

# split in train and test
no_train = int(len(noise_chips)*0.8)
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
