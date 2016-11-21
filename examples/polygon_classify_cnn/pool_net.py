# Generic CNN classifier that uses a geojson file and gbdx imagery to classify chips

import numpy as np
import os, random
import json, geojson

from mltools import geojson_tools as gt
from mltools.data_extractors import get_data_from_polygon_list as get_chips
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD


class PoolNet(object):
    '''
    Convolutional Neural Network model to classify chips as pool/no pool

    INPUT   classes (list [str]): Classes to train model on, exactly as they appear in
                the properties of any geojsons used for training. Defaults to pool
                classes: ['No swimming pool', 'Swimming pool'].
            batch_size (int): Amount of images to use for each batch during training.
                Defaults to 32.
            input_shape (tuple[int]): Shape of input chips with theano dimensional
                ordering (n_channels, height, width). Height and width must be equal. If
                an old model is loaded (old_model_name is not None), input shape will be
                automatically set from the architecture and does not need to be specified.
                Defaults to (3,125,125).
            old_model_name (str): Name of previous model to load (not including file
                extension). There should be a json architecture file and HDF5 ('.h5')
                weights file in the working directory under this name. If None, a new
                model will be compiled for training. Defaults to None.
            learning_rate (float): Learning rate for the first round of training. Defualts
                to 0.001
            small_model (bool): Use a model with nine layers instead of 16. Will train
                faster but may be less accurate and cannot be used with large chips.
                Defaults to False.
            kernel_size (int): Size (in pixels) of the kernels to use at each
                convolutional layer of the network. Defaults to 3 (standard for VGGNet).
    '''

    def __init__(self, classes=['No swimming pool', 'Swimming pool'], batch_size=32,
                 input_shape=(3, 125, 125), small_model=False, model_name=None,
                 learning_rate = 0.001, kernel_size=3):

        self.nb_classes = len(classes)
        self.classes = classes
        self.batch_size = batch_size
        self.small_model = small_model
        self.input_shape = input_shape
        self.lr = learning_rate
        self.kernel_size = kernel_size
        self.cls_dict = {classes[i]: i for i in xrange(len(self.classes))}

        if model_name:
            self.model_name = model_name
            self.model = self._load_model_architecture(model_name)
            self.model.load_weights(model_name + '.h5')
            self.input_shape = self.model.input_shape

        elif self.small_model:
            self.model = self._small_model()

        else:
            self.model = self._VGG_16()

        self.model_layer_names = [self.model.layers[i].get_config()['name']
                                  for i in range(len(self.model.layers))]


    def _VGG_16(self):
        '''
        Implementation of VGG 16-layer net.
        '''
        print 'Compiling VGG Net...'

        model = Sequential()
        model.add(ZeroPadding2D((1,1), input_shape=self.input_shape))
        model.add(Convolution2D(64, self.kernel_size, self.kernel_size,activation='relu',
                                input_shape=self.input_shape))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        sgd = SGD(lr=self.lr, decay=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy')
        return model

    def _small_model(self):
        '''
        Alternative model architecture with fewer layers for computationally expensive
            training datasets
        '''
        print 'Compiling Small Net...'

        model = Sequential()
        model.add(ZeroPadding2D((1,1), input_shape=self.input_shape))
        model.add(Convolution2D(64, self.kernel_size, self.kernel_size,activation='relu',
                                input_shape=self.input_shape))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        sgd = SGD(lr=self.lr, decay=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy')
        return model


    def _load_model_architecture(self, model_name):
        '''
        Load a model arcitecture from a json file

        INPUT   model_name (str): Name of model to load
        OUTPUT  Loaded model architecture
        '''
        print 'Loading model {}'.format(self.model_name)

        #load model
        with open(model_name + '.json') as f:
            mod = model_from_json(json.load(f))

        return mod


    def save_model(self, model_name):
        '''
        Saves model architecture as a json file and current weigts as h5df file

        INPUT   model_name (str): Name inder which to save the architecture and weights.
                    This should not include the file extension.
        '''
        # Save architecture
        arch, arch_json = '{}.json'.format(model_name), self.model.to_json()
        with open(arch, 'w') as f:
            json.dump(arch_json, f)

        # Save weights
        weights = '{}.h5'.format(model_name)
        self.model.save_weights(weights)


    def fit_from_geojson(self, train_geojson, max_side_dim=None, min_side_dim=0,
                         chips_per_batch=5000, train_size=10000, validation_split=0.1,
                         bit_depth=8, save_model=None, nb_epoch=10,
                         shuffle_btwn_epochs=True, return_history=False,
                         save_all_weights=True, retrain=False, learning_rate_2=0.01):
        '''
        Fit a model from a geojson file with training data. This method iteratively
            yields large batches of chips to train on for each epoch. Please ensure that
            your current working directory contains all imagery referenced in the
            image_id property in train_geojson, and are named as follows: <image_id>.tif,
            where image_id is the catalog id of the image.

        INPUT   train_geojson (string): Filename for the training data (must be a
                    geojson). The geojson must be filtered such that all polygons are of
                    valid size (as defined by max_side_dim and min_side_dim)
                max_side_dim (int): Maximum acceptable side dimension (in pixels) for a
                    chip. If None, defaults to input_shape[-1]. If larger than the
                    input shape the chips extracted will be downsampled to match the
                    input shape. Defaults to None.
                min_side_dim (int): Minimum acceptable side dimension (in pixels) for a
                    chip. Defaults to 0.
                chips_per_batch (int): Number of chips to yield per batch. Must be small
                    enough to fit into memory. Defaults to 5000 (decrease for larger
                    input sizes).
                train_size (int): Number of chips to use for training data.
                validation_split (float): Proportion of training chips to use as validation
                    data. Defaults to 0.1.
                bit_depth (int): Bit depth of the image strips from which training chips
                    are extracted. Defaults to 8 (standard for DRA'ed imagery).
                save_model (string): Name of model for saving. if None, does not save
                    model to disk. Defaults to None
                nb_epoch (int): Number of epochs to train for. Each epoch will be trained
                    on batches * batches_per_epoch chips. Defaults to 10.
                shuffle_btwn_epochs (bool): Shuffle the features in train_geojson
                    between each epoch. Defaults to True.
                return_history (bool): Return a list containing metrics from past epochs.
                    Defaults to False.
                save_all_weights (bool): Save model weights after each epoch. A directory
                    called models will be created in the working directory. Defaults to
                    True.
                retrain (bool): Freeze all layers except final softmax to retrain only
                    the final weights of the model. Defaults to False
                learning_rate_2 (float): Learning rate for the second round of training.
                    Only relevant if retrain is True. Defaults to 0.01.
        OUTPUT  trained model, history
        '''
        resize_dim, validation_data, full_hist = None, None, []

        # load geojson training polygons
        with open(train_geojson) as f:
            polygons = geojson.load(f)['features'][:train_size]

        if len(polygons) < train_size:
            raise Exception('Not enough polygons to train on. Please add more training ' \
            'data or decrease train_size.')

        # Determine size of chips to extract and resize dimension
        if not max_side_dim:
            max_side_dim = self.input_shape[-1]

        elif max_side_dim != self.input_shape[-1]:
            resize_dim = self.input_shape # resize chips to match input shape

        # Recompile model with retrain params
        if retrain:
            for i in xrange(len(self.model.layers[:-1])):
                self.model.layers[i].trainable = False

            sgd = SGD(lr=learning_rate_2, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # Set aside validation data
        if validation_split > 0:
            val_size = int(validation_split * train_size)
            val_data, polygons = polygons[: val_size], polygons[val_size: ]
            train_size = len(polygons)

            # extract validation chips
            print 'Getting validation data...\n'
            valX, valY = get_chips(val_data, min_side_dim=min_side_dim,
                                   max_side_dim=max_side_dim, classes=self.classes,
                                   normalize=True, return_labels=True, mask=True,
                                   bit_depth=bit_depth, show_percentage=True,
                                   assert_all_valid=True, resize_dim=resize_dim)

            validation_data = (valX, valY)

        # Train model
        for e in range(nb_epoch):
            print 'Epoch {}/{}'.format(e + 1, nb_epoch)

            # Make callback and diretory for saved weights
            if save_all_weights:
                chk = ModelCheckpoint(filepath="./models/epoch" + str(e) + \
                                      "_{val_loss:.2f}.h5", verbose=1,
                                      save_weights_only=True)

                if 'models' not in os.listdir('.'):
                    os.makedirs('models')

            if shuffle_btwn_epochs:
                np.random.shuffle(polygons)

            # Cycle through batches of chips and train
            for batch_start in range(0, train_size, chips_per_batch):
                callbacks = []
                this_batch = polygons[batch_start: batch_start + chips_per_batch]

                # Get chips from batch
                X, Y = get_chips(this_batch, min_side_dim=min_side_dim,
                                 max_side_dim=max_side_dim, classes=self.classes,
                                 normalize=True, return_labels=True, mask=True,
                                 bit_depth=bit_depth, show_percentage=False,
                                 assert_all_valid=True, resize_dim=resize_dim)

                # Save weights if this is the final batch in the epoch
                if batch_start == range(0, train_size, chips_per_batch)[-1]:
                    callbacks = [chk]

                # Fit the model on this batch
                hist = self.model.fit(X, Y, batch_size=self.batch_size, nb_epoch=1,
                                          validation_data=validation_data,
                                          callbacks=callbacks)

            # Dict recording loss and val_loss after each epoch
            full_hist.append(hist.history)

        if save_model:
            self.save_model(save_model)

        if return_history:
            return full_hist


    def fit_xy(self, X_train, Y_train, validation_split=0.1, save_model=None,
               nb_epoch=10, shuffle_btwn_epochs=True, return_history=False,
               save_all_weights=True, retrain=False, learning_rate_2=0.01):
        '''
        Fit model on training chips already loaded into memory

        INPUT   X_train (array): Training chips with the following dimensions:
                    (train_size, num_channels, rows, cols). Dimensions of each chip
                    should match the input_size to the model.
                Y_train (list): One-hot encoded labels to X_train with dimensions as
                    follows: (train_size, n_classes)
                validation_split (float): Proportion of X_train to validate on while
                    training.
                save_model (string): Name under which to save model. if None, does not
                    save model. Defualts to None.
                nb_epoch (int): Number of training epochs to complete
                shuffle_btwn_epochs (bool): Shuffle the features in train_geojson
                    between each epoch. Defaults to True.
                return_history (bool): Return a list containing metrics from past epochs.
                    Defaults to False.
                save_all_weights (bool): Save model weights after each epoch. A directory
                    called models will be created in the working directory. Defaults to
                    True.
                retrain (bool): Freeze all layers except final softmax to retrain only
                    the final weights of the model. Defaults to False
                learning_rate_2 (float): Learning rate for the second round of training.
                    Only relevant if retrain is True. Defaults to 0.01.
        OUTPUT  trained Keras model.
        '''
        callbacks = []

        # Recompile model with retrain params
        if retrain:
            for i in xrange(len(self.model.layers[:-1])):
                self.model.layers[i].trainable = False

            sgd = SGD(lr=learning_rate_2, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # Define callback to save weights after each epoch
        if save_all_weights:
            chk = ModelCheckpoint(filepath="./models/ch_{epoch:02d}-{val_loss:.2f}.h5",
                                  verbose=1, save_weights_only=True)
            callbacks = [chk]

        # Fit model
        hist = self.model.fit(X_train, Y_train, validation_split=validation_split,
                              callbacks=callbacks, nb_epoch=nb_epoch,
                              shuffle=shuffle_btwn_epochs)

        if save_model:
            self.save_model(save_model)

        if return_history:
            return hist


    def classify_geojson(self, target_geojson, output_name, max_side_dim=None,
                         min_side_dim=0, numerical_classes=True, chips_in_mem=5000,
                         bit_depth=8):
        '''
        Use the current model and weights to classify all polygons in target_geojson. The
            output file will have a 'CNN_class' property with the net's classification
            result, and a 'certainty' property with the net's certainty in the assigned
            classification.
        Please ensure that your current working directory contains all imagery referenced
            in the image_id property in target_geojson, and are named as follows:
            <image_id>.tif, where image_id is the catalog id of the image.

        INPUT   target_geojson (string): Name of the geojson to classify. This file
                    should only contain chips with side dimensions between min_side_dim
                    and max_side_dim (see below).
                output_name (string): Name under which to save the classified geojson.
                max_side_dim (int): Maximum acceptable side dimension (in pixels) for a
                    chip. If None, defaults to input_shape[-1]. If larger than the
                    input shape the chips extracted will be downsampled to match the
                    input shape. Defaults to None.
                min_side_dim (int): Minimum acceptable side dimension (in pixels) for a
                    chip. Defaults to 0.
                numerical_classes (bool): Make output classifications correspond to the
                    indicies (base 0) of the 'classes' attribute. If False, 'CNN_class'
                    is a string with the class name. Defaults to True.
                chips_in_mem (int): Number of chips to load in memory at once. Decrease
                    this parameter for larger chip sizes. Defaults to 5000.
                bit_depth (int): Bit depth of the image strips from which training chips
                    are extracted. Defaults to 8 (standard for DRA'ed imagery).

        '''
        resize_dim, yprob, ytrue = None, [], []

        # Determine size of chips to extract and resize dimension
        if not max_side_dim:
            max_side_dim = self.input_shape[-1]

        elif max_side_dim != self.input_shape[-1]:
            resize_dim = self.input_shape # resize chips to match input shape

        # Format output filename
        if not output_name.endswith('.geojson'):
            output_name = '{}.geojson'.format(output_name)

        # Get polygon list from geojson
        with open(target_geojson) as f:
            features = geojson.load(f)['features']

        # Classify in batches of 1000
        for ix in xrange(0, len(features), chips_in_mem):
            this_batch = features[ix: (ix + chips_in_mem)]
            try:
                X = get_chips(this_batch, min_side_dim=min_side_dim,
                              max_side_dim=max_side_dim, classes=self.classes,
                              normalize=True, return_labels=False,
                              bit_depth=bit_depth, mask=True, show_percentage=False,
                              assert_all_valid=True, resize_dim=resize_dim)

            except (AssertionError):
                raise ValueError('Please filter the input geojson file using ' \
                                 'geojoson_tools.filter_geojson() and ensure all ' \
                                 'polygons are valid before using this method.')

            # Predict classes of test data
            yprob += list(self.model.predict_proba(X))

        # Get predicted classes and certainty
        yhat = [np.argmax(i) for i in yprob]
        ycert = [str(np.max(j)) for j in yprob]

        if not numerical_classes:
            yhat = [self.classes[i] for i in yhat]

        # Update geojson, save as output_name
        data = zip(yhat, ycert)
        property_names = ['CNN_class', 'certainty']
        gt.write_properties_to(data, property_names=property_names,
                               input_file=target_geojson, output_file=output_name)



# Tools for analyzing network performance

def x_to_rgb(X):
    '''
    Transform a normalized (3,h,w) image (theano ordering) to a (h,w,3) rgb image
    (tensor flow).
    Use this to view or save rgb polygons as images.

    INPUT   (1) 3d array 'X': originial chip with theano dimensional ordering (3, h, w)
    OUTPUT  (1) 3d array: rgb image in tensor flow dim-prdering (h,w,3)
    '''

    rgb_array = np.zeros((X.shape[1], X.shape[2], 3), 'uint8')
    rgb_array[...,0] = X[0] * 255
    rgb_array[...,1] = X[1] * 255
    rgb_array[...,2] = X[2] * 255
    return rgb_array
