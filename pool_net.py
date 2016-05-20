import numpy as np
import random
import json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.models import Sequential, Graph, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

# input_size = (224,224,3)
# nb_classes = 2
#
# model = Sequential()
#
# model.add(Convolution2D(96, 11, 11, border_mode = 'valid', dim_ordering = 'tf', input_shape = input_size, subsample = (4,4), activation = 'relu'))
# model.add(BatchNormalization(mode=0, axis=-1))
# model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))
#
# model.add(Convolution2D(256, 5, 5, border_mode = 'valid', activation = 'relu'))
# model.add(BatchNormalization(mode=0, axis=-1))
# model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))
#
# model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))
#
# model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))
#
# model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))
#
# model.add(Flatten())
# model.add(Dense(2048))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2048))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
#
# sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9)
#
# model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')


class PoolNet(object):
    '''
    Fully Convolutional model to classify polygons as pool/no pool
    '''

    def __init__(self, nb_chan=3, nb_epoch=4, nb_classes=2, batch_size=32, input_size=(3, 224, 224), n_dense_nodes = 2048):
        self.nb_epoch = nb_epoch
        self.nb_chan = nb_chan
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_dense_nodes = n_dense_nodes
        self.model = self.compile_model()
        self.model_layer_names = [self.model.layers[i].get_config()['name'] for i in xrange(len(self.model.layers))]

    def compile_model(self):
        print 'Compiling model...'
        model = Sequential()

        model.add(Convolution2D(96, 11, 11, border_mode = 'valid', input_shape = input_size, activation = 'relu'))

        model.add(Convolution2D(256, 5, 5, border_mode = 'valid', activation = 'relu'))
        model.add(BatchNormalization(mode=0, axis=-1))
        model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))

        model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))
        model.add(BatchNormalization(mode=0, axis=-1))
        model.add(MaxPooling2D(pool_size = (3,3), strides=(2,2)))

        model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))

        model.add(Convolution2D(256, 3, 3, border_mode = 'valid', activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))

        model.add(Flatten())
        model.add(Dense(self.n_dense_nodes))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_dense_nodes))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9)

        model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')

        return model

    def get_behead_index(self, layer_names):
        '''
        INPUT   (1) list 'layer_names': names of each layer in model
        '''
        for i, layer_name in enumerate(layer_names):
            if i > 0 and layer_name[:7] == 'flatten':
                if layer_names[i-1][:7] != 'dropout':
                    behead_ix = i
                else:
                    behead_ix = i - 1
        return behead_ix

    def make_fc_model(self):
        # get index of first dense layer in model
        behead_ix = self.get_behead_index(self.model_layer_names)
        model_layers = self.model.layers[:behead_ix]

        # replace dense layers with convolutions
        model = Sequential()
        model_layers += [Convolution2D(self.n_dense_nodes, 1, 1)]
        model_layers += [Activation('relu')]
        model_layers += [Convolution2D(self.n_dense_nodes, 1, 1)]
        model_layers += [Activation('relu')]
        model_layers += [Convolution2D(self.nb_classes, 1, 1)]
        # import pdb; pdb.set_trace()
        model_layers += [Reshape((1, self.nb_classes))]
        model_layers += [Activation('softmax')]

        for process in model_layers:
            model.add(process)
        return model
