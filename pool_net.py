import numpy as np
import random
import json
from polygon_pipeline import get_iter_data
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.models import Sequential, Graph, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

class PoolNet(object):
    '''
    Fully Convolutional model to classify polygons as pool/no pool
    INPUT   (1) int 'nb_chan': number of input channels. defaults to 3 (rgb)
            (2) int 'nb_epoch': number of epochs to train. defaults to 4
            (3) int 'nb_classes': number of different image classes. defaults to 2 (pool/no pool)
            (4) int 'batch_size': amount of images to train for each batch. defaults to 32
            (5) list[int] 'input_shape': shape of input images (3-dims). defaults to (3,224,224)
            (6) int 'n_dense_nodes': number of nodes to use in dense layers. defaults to 2048.
            (7) bool 'fc': True for fully convolutional model, else classic convolutional model. defaults to True
    '''

    def __init__(self, nb_chan=3, nb_epoch=4, nb_classes=2, batch_size=32, input_shape=(3, 224, 224), n_dense_nodes = 2048, fc = True):
        self.nb_epoch = nb_epoch
        self.nb_chan = nb_chan
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_dense_nodes = n_dense_nodes
        self.fc = fc
        self.model = self.compile_model()
        self.model_layer_names = [self.model.layers[i].get_config()['name'] for i in xrange(len(self.model.layers))]
        if self.fc:
            self.fc_model = self.make_fc_model()

    def compile_model(self):
        '''
        compiles standard convolutional netowrk (not FCNN)
        '''
        print 'Compiling standard model...'
        model = Sequential()

        model.add(Convolution2D(96, 11, 11, border_mode = 'valid', input_shape = self.input_shape, activation = 'relu'))

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
        helper function to find index where net flattens
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
        '''
        creates a fully convolutional model from self.model
        '''
        # get index of first dense layer in model
        behead_ix = self.get_behead_index(self.model_layer_names)
        model_layers = self.model.layers[:behead_ix]

        # replace dense layers with convolutions
        model = Sequential()
        model_layers += [Convolution2D(self.n_dense_nodes, 1, 1)]
        model_layers += [Activation('relu')]
        model_layers += [Convolution2D(self.n_dense_nodes/2, 1, 1)]
        model_layers += [Activation('relu')]
        model_layers += [Convolution2D(self.nb_classes, 22, 22)]
        # import pdb; pdb.set_trace()
        model_layers += [Reshape((self.nb_classes,1))]
        model_layers += [Activation('softmax')]

        print 'Compiling Fully Convolutional Model...'
        for process in model_layers:
            model.add(process)
        print 'Done.'
        return model

    def train_on_data(self, shapefile, fc_model=True):
        '''
        Uses generator to train model from shapefile

        INPUT   (1) string 'shapefile': geojson file containing polygons to be trained on
                (2) bool 'fc_model': True to use fully convolutional model
        OUTPUT  (1) trained model
        '''
        print 'Training model on batches...'

        if self.fc:
            mod = self.fc_model
        else:
            mod = self.model

        for epoch in xrange(self.nb_epoch):
            print 'Epoch {}:'.format(epoch)

            for chips, ids, labels in get_iter_data(shapefile, batch_size=self.batch_size, min_chip_hw=20, max_chip_hw=50, return_labels=True, mask=True):
                y = [1 if label == 'Swimming pool' else 0 for label in labels]
                Y = np_utils.to_categorical(y, self.nb_classes)

                print 'Training...'
                import pdb; pdb.set_trace()
                mod.train_on_batch(chips, Y)

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
