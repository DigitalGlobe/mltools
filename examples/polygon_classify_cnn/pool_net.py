# Generic CNN classifier that uses a geojson file and gbdx imagery to classify polygons

import numpy as np
import os
import random
import json
import geojson
import subprocess
from mltools import data_extractors as de
from mltools.data_extractors import get_iter_data, getIterData
from mltools.geojson_tools import write_properties_to, filter_polygon_size
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation, Flatten, Reshape
from keras.models import Sequential, Graph, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1l2
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import classification_report
from time import localtime

class PoolNet(object):
    '''
    Convolutional Neural Network model to classify polygons as pool/no pool

    INPUT   classes (list): classes to train images on, exactly as they appear in
                shapefile properties. defaults to pool classes "['No swimming pool',
                'Swimming pool']"
            max_chip_hw (int): maximum acceptable side dimension for polygons. Defaults
                to 125
            min_chip_hw (int): minimum acceptable side dimension for polygons. Defaults
                to 0 (will not exclude chips deemed too small).
            batch_size (int): amount of images to use for each batch during training.
                defaults to 32.
            input_shape (tuple[int]): shape of input images (3-dims). defaults to
                (3,125,125)
            fc (bool): True for fully convolutional model, else classic convolutional
                model. defaults to False.
            old_model (bool): Use a saved trained model (model_name) architecture and
                weights. Defaults to False
            model_name (str): Only relevant if load_model is True. name of model (not
                including file extension) to load. Defaults to None
            learning_rate (float): learning rate for the first round of training. Defualts
                to 0.001
            bit_depth (int): bit depth of the imagery trained on. Used for normalization
                of chips. Defaults to 11.
            kernel_size (int): size (in pixels) of the kernels to use at each
                convolutional layer of the network. Defaults to 3 (standard for VGGNet)
    '''

    def __init__(self, classes=['No swimming pool', 'Swimming pool'], max_chip_hw=125,
                min_chip_hw=0, batch_size=32, input_shape=(3, 125, 125), fc=False,
                old_model=False, small_model=False, model_name=None, learning_rate = 0.001,
                bit_depth=11, kernel_size=3):

        self.nb_classes = len(classes)
        self.classes = classes
        self.max_chip_hw = max_chip_hw
        self.min_chip_hw = min_chip_hw
        self.batch_size = batch_size
        self.fc = fc
        self.old_model = old_model
        self.small_model = small_model
        self.input_shape = input_shape
        self.lr = learning_rate
        self.bit_depth = bit_depth
        self.kernel_size = kernel_size
        self.cls_dict = {classes[i]: i for i in xrange(len(self.classes))}

        if self.old_model:
            self.model_name = model_name
            self.model = self.load_model(model_name)
            self.model.load_weights(model_name + '.h5')
            self.max_side_dim = self.model.input_shape[-1]
            self.nb_classes = self.model.output_shape[-1]
            self.input_shape = (self.model.input_shape[1], self.max_side_dim,
                                self.max_side_dim)

        elif self.small_model:
            self.model = self._small_model()

        else:
            self.model = self._VGG_16()

        self.model_layer_names = [self.model.layers[i].get_config()['name']
                                    for i in range(len(self.model.layers))]
        if self.fc:
            self.model = self.make_fc_model()

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


    def fit_from_geojson(self, train_shapefile, chips_per_batch = 5000,
                         batches_per_epoch=4, validation_split=0.1, save_model=None,
                         nb_epoch=10, shuffle_btwn_epochs=True, return_history=False,
                         save_all_weights=True, retrain=False, lr_2=0.01, resize_dim=None):
        '''
        Fit a model using a generator that iteratively yields large batches of chips to
            train on for each epoch.
        INPUT   train_shapefile (string): Filename for the training data (must be a
                    geojson). The geojson must be filtered such that all polygons are of
                    valid size (as defined by max_side_dim and min_side_dim)
                chips_per_batch (int): Number of chips to yield per batch. Must be small
                    enough to fit into memory. Defaults to 5000.
                batches_per_epoch (int): number of batches two train on per epoch. Notice
                    that the train size will be equal to chips_per_batch *
                    batches_per_epoch. Defualts to 4.
                validation_split (float): proportion of chips to use as validation data.
                    Defaults to 0.1.
                save_model (string): name of model for saving. if None, does not save
                    model. Defaults to None
                nb_epoch (int): Number of epochs to train for. Each epoch will be trained
                    on batches * batches_per_epoch chips. Defaults to 10.
                shuffle_btwn_epochs (bool): Shuffle the features in train_shapefile
                    between each epoch. Defaults to True.
                return_history (bool): Return a list containing metrics from past epochs.
                    Defaults to False.
                save_all_weights (bool): Save model weights after each epoch. A directory
                    called models will be created in the working directory. Defaults to
                    True.
                retrain (bool): freeze all layers except final softmax to retrain only
                    the final weights of the model. Defaults to False
                lr_2 (float): Learning rate for the second round of training. Only
                    relevant if retrain is True. Defaults to 0.01.
        OUTPUT  trained model, history
        '''

        train_size = chips_per_batch * batches_per_epoch
        full_hist = []

        if retrain:
            # freeze all layers except final dense
            for i in xrange(len(self.model.layers[:-1])):
                self.model.layers[i].trainable = False

            # recompile model
            sgd = SGD(lr=lr_2, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # load geojson polygons
        with open(train_shapefile) as f:
            polygons = geojson.load(f)['features'][:train_size]

        # check for sufficient training data
        if len(polygons) < train_size:
            raise Exception('Not enough polygons to train on. Please add more training' \
                            ' data or decrease value of batches_per_epoch.')

        # set aside validation data
        if validation_split > 0:
            val_size = int(validation_split * train_size)
            val_data = polygons[: val_size]
            polygons = polygons[val_size: ]
            chips_per_batch = int((1 - validation_split) * chips_per_batch)

            # extract validation chips
            print 'Getting validation data...\n'
            valX, valY = de.get_data_from_polygon_list(val_data, min_chip_hw=self.min_chip_hw,
                                                       max_chip_hw=self.max_chip_hw,
                                                       classes=self.classes, normalize=True,
                                                       return_labels=True,
                                                       bit_depth=self.bit_depth, mask=True,
                                                       show_percentage=False,
                                                       assert_all_valid=True,
                                                       resize_dim=resize_dim)

        for e in range(nb_epoch):
            # make diretory for saved weights
            if save_all_weights:
                chk = ModelCheckpoint(filepath="./models/epoch" + str(e) + \
                                      "_{val_loss:.2f}.h5",
                                      verbose=1, save_weights_only = True)
                if 'models' not in os.listdir('.'):
                    subprocess.call('mkdir models', shell=True)

            print 'Epoch {}/{}'.format(e + 1, nb_epoch)

            # shuffle data to randomize net input
            if shuffle_btwn_epochs:
                np.random.shuffle(polygons)

            for batch in range(batches_per_epoch):
                curr_ix = batch * chips_per_batch
                this_batch = polygons[curr_ix: curr_ix + chips_per_batch]

                # extract chips from each batch to train on
                X, Y = de.get_data_from_polygon_list(this_batch,
                                                     min_chip_hw=self.min_chip_hw,
                                                     max_chip_hw=self.max_chip_hw,
                                                     classes=self.classes, normalize=True,
                                                     return_labels=True,
                                                     bit_depth=self.bit_depth, mask=True,
                                                     show_percentage=False,
                                                     assert_all_valid=True,
                                                     resize_dim=resize_dim)

                # train with validation data
                if validation_split > 0:
                    if save_all_weights and batch == batches_per_epoch - 1:
                        hist = self.model.fit(X, Y, batch_size=self.batch_size, nb_epoch=1,
                                              validation_data=(valX, valY),
                                              callbacks=[chk])

                    else:
                        hist = self.model.fit(X, Y, batch_size=self.batch_size, nb_epoch=1,
                                              validation_data=(valX, valY))

                # train without validation data
                else:
                    if save_all_weights and batch == batches_per_epoch - 1:
                        chk = ModelCheckpoint(filepath="./models/epoch" + str(e) + ".h5",
                                              verbose=1)
                        hist = self.model.fit(X, Y, batch_size=self.batch_size, nb_epoch=1,
                                              callbacks=[chk])

                    else:
                        hist = self.model.fit(X, Y, batch_size=self.batch_size, nb_epoch=1)

            # dict recording loss and val_loss after each epoch
            full_hist.append(hist.history)

        if save_model:
            self.save_model(save_model)

        if return_history:
            return full_hist

    def fit_xy(self, X_train, Y_train, validation_split=0.1,
                      save_model = None, nb_epoch=15):
        '''
        Fit model on training chips already loaded into memory

        INPUT   X_train (array): Training chips with the following dimensions:
                    (train_size, num_channels, rows, cols)
                Y_train (list): One-hot encoded labels to X_train with dimenstions as
                    follows: (train_size, n_classes)
                validation_split (float): Proportion of X_train to validate on while
                    training.
                save_model (string): Name under which to save model. if None, does not
                    save model. Defualts to None.
                nb_epoch (int): Number of training epochs to complete
        OUTPUT  trained Keras model.
        '''
        # Define callback to save weights after each epoch
        checkpointer = ModelCheckpoint(filepath="./models/ch_{epoch:02d}-{val_loss:.2f}.h5",
                                       verbose=1, save_weights_only=True)

        self.model.fit(X_train, Y_train, validation_split=validation_split,
                       callbacks=[checkpointer], nb_epoch=nb_epoch)

        if save_model:
            self.save_model(save_model)


    def retrain_output(self, X_train, Y_train, learning_rate=0.01, **kwargs):
        '''
        Retrains last dense layer of model on chips loaded into memory. For use with
            unbalanced classes after training on balanced data.
        INPUT   X_train(array): Training chips with the following dimensions:
                    (train_size, num_channels, rows, cols)
                Y_train (list): One-hot encoded labels to X_train with dimenstions as
                    follows: (train_size, n_classes)
                learning_rate (float): Learning rate
                validation_split (float): Proportion of X_train to validate on while
                    training.
                save_model (string): Name under which to save model. if None, does not
                    save model. Defualts to None.
                nb_epoch (int): Number of training epochs to complete
        OUTPUT  (1) retrained model
        '''
        # freeze all layers except final dense
        for i in xrange(len(self.model.layers[:-1])):
            self.model.layers[i].trainable = False

        # recompile model
        sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # train model
        self.fit_xy(X_train, Y_train, **kwargs)



    def save_model(self, model_name):
        '''
        INPUT   (1) string 'model_name': name to save model and weigths under, including
        filepath but not extension
        Saves current model as json and weigts as h5df file
        '''
        model = '{}.json'.format(model_name)
        weights = '{}.h5'.format(model_name)
        log = '{}.txt'.format(model_name)
        json_string = self.model.to_json()
        self.model.save_weights(weights)
        with open(model, 'w') as f:
            json.dump(json_string, f)

        # make log for model train
        time = localtime()
        date = str(time[1]) + '-' + str(time[2]) + '-' + str(time[0]) + '\n' + \
        str(time[3]) + ':' + str(time[4]) + ':' + str(time[5]) + '\n'
        layers = str(self.model.layers)
        with open(log, 'w') as l:
            l.write(date + layers)


    def load_model(self, model_name):
        '''
        INPUT  (1) string 'model_name': filepath to model
        OUTPUT: Loaded model architecture
        '''
        print 'Loading model {}'.format(self.model_name)

        #load model
        with open(model_name + '.json') as f:
            mod = model_from_json(json.load(f))
        return mod


    def classify_geojson(self, geoj, output_name, numerical_classes=True,
                         resize_dim=None, img_name=None):
        '''
        Use the current model and weights to classify all polygons (of appropriate size)
        in the given geojson. Records PoolNet classification, whether or not it was
        misclassified by PoolNet, and the certainty for the given class.
        INPUT   geoj (string): name of the geojson to classify
                output_name (string): name to give the classified geojson
                numerical_classes (bool): make output classifications numbers instead of
                    strings. If False, class number will be used as the index to the
                    classes argument (class 0 = self.classes[0]). Defaults to True.
                resize_dim (tuple): Dimensions to resize image to.
                image_name (string): name of the associated geotiff image if different
                    than catalog number. Defaults to None
        '''

        yprob, ytrue = [], []
        if output_name.endswith('.geojson'):
            output_file = output_name
        else:
            output_file = '{}.geojson'.format(output_name)

        # Open get chips from geojson
        with open(geoj) as f:
            features = geojson.load(f)['features']

        # Classeify in batches of 1000
        for ix in xrange(0, len(features), 1000):
            this_batch = features[ix: (ix + 1000)]
            try:
                X = de.get_data_from_polygon_list(this_batch,
                                                  min_chip_hw=self.min_chip_hw,
                                                  max_chip_hw=self.max_chip_hw,
                                                  classes=self.classes, normalize=True,
                                                  return_labels=False,
                                                  bit_depth=self.bit_depth, mask=True,
                                                  show_percentage=False,
                                                  assert_all_valid=True,
                                                  resize_dim=resize_dim)
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
        write_properties_to(data, property_names=property_names, input_file=geoj,
                            output_file=output_file)


    def _get_behead_index(self, layer_names):
        '''
        helper function to find index where net flattens
        INPUT   (1) list 'layer_names': names of each layer in model
        OUTPUT  (1) int 'behead_ix': index of flatten layer
        '''
        for i, layer_name in enumerate(layer_names):
            # Find first dense layer, remove preceeding dropout if applicable
            if i > 0 and layer_name[:7] == 'flatten':
                if layer_names[i-1][:7] != 'dropout':
                    behead_ix = i
                else:
                    behead_ix = i - 1
        return behead_ix

    def _get_val_data(self, shapefile, val_size):
        '''
        hacky... don't use for actual training purposes.
        creates validation data from input shapefile to use with fit_generator function
        '''
        # shuffle features in orig shapefile, use for val data
        with open(shapefile) as f:
            data = geojson.load(f)
            feats = data['features']
            np.random.shuffle(feats)
            data['features'] = feats

        with open('tmp_val.geojson', 'w') as f:
            geojson.dump(data, f)

        val_gen = getIterData('tmp_val.geojson', batch_size=val_size,
                              min_chip_hw=self.min_chip_hw, max_chip_hw=self.max_chip_hw,
                              classes=self.classes, bit_depth=self.bit_depth,
                              show_percentage=False)

        x, y = val_gen.next()
        subprocess.call('rm tmp_val.geojson', shell=True)
        return x, y

    def make_fc_model(self):
        '''
        creates a fully convolutional model from self.model
        '''
        # get index of first dense layer in model
        behead_ix = self._get_behead_index(self.model_layer_names)
        model_layers = self.model.layers[:behead_ix]
        # shape of image entering FC layers
        inp_shape = self.model.layers[behead_ix - 1].get_output_shape_at(-1)

        # replace dense layers with convolutions
        model = Sequential()
        model_layers += [Convolution2D(2048, 1, 1)]
        model_layers += [Activation('relu')]
        model_layers += [Convolution2D(2048, 1, 1)]
        model_layers += [Activation('relu')]
        model_layers += [Convolution2D(self.nb_classes, inp_shape[-1], inp_shape[-1])]
        # must be same shape as target vector (None, num_classes, 1)
        model_layers += [Reshape((self.nb_classes-1,1))]
        model_layers += [Activation('softmax')]

        print 'Compiling Fully Convolutional Model...'
        for process in model_layers:
            model.add(process)
        sgd = SGD(lr=self.lr, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return model


    def fit_generator(self, train_shapefile, train_size=10000, save_model=None,
                      nb_epoch=5, validation_prop=0.1):
        '''
        Fit a model using a generator that yields a large batch of chips to train on.
        INPUT   (1) string 'train_shapefile': filename for the training data (must be a
                    geojson)
                (2) int 'train_size': number of chips to train model on. Defaults to 10000
                (3) string 'save_model': name of model for saving. if None, does not
                    save model.
                (4) int 'nb_epoch': Number of epochs to train for
                (5) float 'validation_prop': proportion of training data to use for
                    validation. defaults to 0.1. Does not do validation if validation_prop
                    is None.
        OUTPUT  (1) trained model.
        '''
        # es = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
        data_gen = getIterData(train_shapefile, batch_size=self.batch_size,
                               min_chip_hw=self.min_chip_hw, max_chip_hw=self.max_chip_hw,
                               classes=self.classes, bit_depth=self.bit_depth, cycle=True,
                               show_percentage=False)

        if validation_prop:
            checkpointer = ModelCheckpoint(filepath="./models/epoch_{epoch:02d}-{val_loss:.2f}.h5",
                                           verbose=1, save_weights_only=True)
            valX, valY = self._get_val_data(train_shapefile,
                                            int(validation_prop * train_size))

            self.model.fit_generator(data_gen, samples_per_epoch=train_size,
                                     nb_epoch=nb_epoch, callbacks=[checkpointer],
                                     validation_data = (valX, valY))

        else:
            checkpointer = ModelCheckpoint(filepath="./models/epoch_{epoch:02d}-{loss:.2f}.h5",
                                           verbose=1, save_weights_only=True)
            self.model.fit_generator(data_gen, samples_per_epoch=train_size,
                                     nb_epoch=nb_epoch, callbacks=[checkpointer])

        if save_model:
            self.save_model(save_model)


    def retrain_output_on_generator(self, train_shapefile, retrain_size=5000,
                                    learning_rate=0.01, save_model=None, nb_epoch=5,
                                    validation_prop=0.1):
        '''
        Retrains last dense layer of model with a generator. For use with unbalanced
        classes after training on balanced data.
        INPUT   (1) string 'train_shapefile': filename for the training data (must be a
                    geojson)
                (2) int 'train_size': number of chips to train model on. Defaults to 5000
                (3) float 'lr'
                (4) string 'save_model': name of model for saving. if None, does not
                    save model.
                (5) int 'nb_epoch': Number of epochs to train for. Defaults to 5
                (6) float 'validation_prop': proportion of training data to use for
                    validation. defaults to 0.1. Does not do validation if validation_prop
                    is None.

        OUTPUT  (1) retrained model.
        '''
        # freeze all layers except final dense
        for i in xrange(len(self.model.layers[:-1])):
            self.model.layers[i].trainable = False

        # recompile model
        sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # train model with frozen weights
        data_gen = getIterData(train_shapefile, batch_size=self.batch_size,
                               min_chip_hw=self.min_chip_hw, max_chip_hw=self.max_chip_hw,
                               classes=self.classes, bit_depth=self.bit_depth, cycle=True,
                               show_percentage=False)

        if validation_prop:
            checkpointer = ModelCheckpoint(filepath="./models/epoch_{epoch:02d}-{val_loss:.2f}.h5",
                                           verbose=1, save_weights_only=True)
            valX, valY = self._get_val_data(train_shapefile,
                                            int(validation_prop * retrain_size))

            self.model.fit_generator(data_gen, samples_per_epoch=retrain_size,
                                     nb_epoch=nb_epoch, callbacks=[checkpointer],
                                     validation_data=(valX, valY))

        else:
            checkpointer = ModelCheckpoint(filepath="./models/epoch_{epoch:02d}-{loss:.2f}.h5",
                                           verbose=1, save_weights_only=True)
            self.model.fit_generator(data_gen, samples_per_epoch=retrain_size)

        if save_model:
            self.save_model(save_model)

    def evaluate_model(self, X_test, Y_test, return_yhat=False):
        '''
        Predicts classes of X_test and evaluates precision, recall and f1 score
        INPUT   (1) array 'X_test': array of chips
                (2) list 'y_test': labels corresponding to chips in X_test
                (3) bool 'return_yhat': return the values of predicted classes for X_test
        OUTPUT  (1) classification report
        '''
        # classify chips from trained net
        y_hat = self.model.predict_classes(X_test)
        y_true = [int(i[1]) for i in Y_test]
        print classification_report(y_true, y_hat)

        if return_yhat:
            return y_hat


# Evaluation methods

def confusion_matrix_imgs(X_test, y_test, y_pred):
    '''
    Generate file with incorrectly classified polygons for inspection
    INPUT   (1) array 'X_test': array of chips
            (2) list 'y_test': labels corresponding to chips in X_test
            (3) list 'y_pred': results of classification on X_test. if None, will
            classify X_test and generate y_pred.
    OUTPUT  (1) true positives
            (2) true negatives
            (3) false positives
            (4) false negatives
    '''
    wrong = X_test[[y_test!=y_pred]]
    # Find chips yieling a false positive and false negative
    fp, fn = wrong[[y_test==0]], wrong[[y_test==1]]

    right = X_test[[y_test==y_pred]]
    # Find chips yielding a true positive and true negative
    tp, tn = right[[y_test==1]], right[[y_test==0]]

    return tp, tn, fp, fn


def x_to_rgb(X):
    '''
    Transform a normalized (3,h,w) image (theano ordering) to a (h,w,3) rgb image
    (tensor flow).
    Use this when viewing polygons as a color image in matplotlib.

    INPUT   (1) 3d array 'X': originial chip with theano dimensional ordering (3, h, w)
    OUTPUT  (1) 3d array: rgb image in tensor flow dim-prdering (h,w,3)
    '''
    rgb_array = np.zeros((X.shape[1], X.shape[2], 3), 'uint8')
    rgb_array[...,0] = X[0] * 255
    rgb_array[...,1] = X[1] * 255
    rgb_array[...,2] = X[2] * 255
    return rgb_array


def filter_by_classification(shapefile, output_name, max_cert=0.75, min_cert=0.5,
                             missed=True):
    '''
    Method for filtering a geojson file by max and min classification certainty.
    INPUT   (1) string 'input_file': name of shapefile to filter. should be output of
            classify_shapefile method from PoolNet.
            (2) string 'output_name': name of file (not including extension) to save
            filtered geojson to.
            (3) float 'max_cert': Maximum acceptable certainty from PoolNet classification
            (4) float 'min_cert': Maximum acceptable certainty from PoolNet classification
            (5) bool 'missed': use only misclassfied polygons. Defaults to True
    OUTPUT  (1) shapefile filtered by certainty and missed.
    '''

    with open(shapefile) as f:
        data = geojson.load(f)
    ok_polygons = []

    # Cycle through each polygon, check for certainty and misclassification
    print 'Filtering polygons...'
    for geom in data['features']:
        try:
            cert = geom['properties']['certainty']
            if geom['properties']['missed'] == 0:
                misclass = False
            else:
                misclass = True
            # gt = geom['properties']['class_name']
        except (KeyError):
            continue

        if misclass == missed:
            if cert >= min_cert and cert <= max_cert:
                ok_polygons.append(geom)

    # Save filtered polygons to nex shapefile
    filtrate = {data.keys()[0]: data.values()[0], data.keys()[1]: ok_polygons}
    with open('{}.geojson'.format(output_name), 'wb') as f:
        geojson.dump(filtrate, f)

    print 'Saved {} polygons as {}.geojson'.format(len(ok_polygons), output_name)
