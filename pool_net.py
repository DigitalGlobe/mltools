import numpy as np
import random
import json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential, Graph, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

class PoolNet(object):
    '''
    Fully Convolutional model to classify polygons as pool/no pool
    
    '''

nb_epoch = 5
n_chan = 3
batch_size = 32
input_size = (224,224)

model = Sequential()

model.add(Convolution2D(96, 11, 11, border_mode = 'valid', dim_ordering = 'tf', input_shape = input_size, subsample = (4,4), activation = 'relu'))
model.add(BatchNormalization(mode=0, axis=-1))
model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))

model.add(Convolution2D(256, 5, 5, border_mode = 'valid', activation = 'relu'))
model.add(BatchNormalization(mode=0, axis=-1))
model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2))

model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))

model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))

model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))

model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9)

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')
