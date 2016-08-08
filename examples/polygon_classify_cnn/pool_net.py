import numpy as np
import random
import json
import geojson
import subprocess
from mltools.data_extractors import get_iter_data, getIterData
from mltools.geojson_tools import write_properties_to
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
            lr_1 (float): learning rate for the first round of training. Defualts to
                0.001
            lr_2 (float): learning rate for second round of training (just output layer).
                Defaults to 0.01
    '''

    def __init__(self, classes=['Swimming pool', 'No swimming pool'], max_chip_hw=125,
                min_chip_hw=0, batch_size=32, input_shape=(3, 125, 125), fc = False,
                old_model=False, model_name=None, lr_1 = 0.001, lr_2 = 0.01):

        self.nb_classes = len(classes)
        self.classes = classes
        self.max_chip_hw = max_chip_hw
        self.min_chip_hw = min_chip_hw
        self.batch_size = batch_size
        self.fc = fc
        self.old_model = old_model
        self.input_shape = input_shape
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.cls_dict = {classes[i]: i for i in xrange(len(self.classes))}

        if self.old_model:
            self.model_name = model_name
            self.model = self.load_model(model_name)
            self.model.load_weights(model_name + '.h5')
            self.max_side_dim = self.model.input_shape[-1]
            self.input_shape = (self.model.input_shape[1], self.max_side_dim,
                                self.max_side_dim)
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
        model.add(Convolution2D(64, 3, 3,activation='relu',input_shape=self.input_shape))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        sgd = SGD(lr=self.lr_1, decay=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy')
        return model

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
        hacky...
        creates validation data from input shapefile to use with fit_generator function
        '''
        with open(shapefile) as f:
            data = geojson.load(f)
            feats = data['features']
            np.random.shuffle(feats)
            data['features'] = feats

        with open('tmp_val.geojson', 'w') as f:
            geojson.dump(data, f)

        val_gen = getIterData('tmp_val.geojson', batch_size=val_size,
                              min_chip_hw=self.min_chip_hw, max_chip_hw=self.max_chip_hw,
                              classes=self.classes)

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
        sgd = SGD(lr=self.lr_1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return model


    def fit_xy(self, X_train, Y_train, validation_split=0.1,
               save_model = None, nb_epoch=15):
        '''
        Fit model on pre-loaded training data. Only for sizes small enough to fit in
        memory
        INPUT   (1) array 'X_train': training chips in the shape (train_size, 3, h, w)
                (2) list 'Y_train': one-hot associated labels to X_train. shape =
                train_size, n_classes)
                (3) float 'validation_split': proportion of X_train to validate on.
                (4) string 'save_model': name of model for saving. if None, does not
                save model.
                (5) int 'nb_epoch': Number of epochs to train for
        OUTPUT  (1) trained model.
        '''
        # Define callback to save weights after each epoch
        # es = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
        checkpointer = ModelCheckpoint(filepath="./models/ch_{epoch:02d}-{val_loss:.2f}.h5",
                                       verbose=1)

        self.model.fit(X_train, Y_train, validation_split=validation_split,
                       callbacks=[checkpointer], nb_epoch=nb_epoch)

        if save_model:
            self.save_model(save_model)


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
        OUTPUT  (1) trained model.
        '''
        # es = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
        checkpointer = ModelCheckpoint(filepath="./models/ch_{epoch:02d}-{loss:.2f}.h5",
                                       verbose=1)

        data_gen = getIterData(train_shapefile, batch_size=self.batch_size,
                               min_chip_hw=self.min_chip_hw, max_chip_hw=self.max_chip_hw,
                               classes=self.classes)

        if validation_prop:
            valX, valY = self._get_val_data(train_shapefile,
                                            int(validation_prop * train_size))

            self.model.fit_generator(data_gen, samples_per_epoch=train_size,
                                     nb_epoch=nb_epoch, callbacks=[checkpointer],
                                     validation_data = (valX, valY))

        else:
            self.model.fit_generator(data_gen, samples_per_epoch=train_size,
                                     nb_epoch=nb_epoch, callbacks=[checkpointer])

        if save_model:
            self.save_model(save_model)

    def fit_with_augmentation(self, train_shapefile, chips_to_yield, train_size,
                              nb_epoch=10, validation_prop=0.1, save_model=None):
        '''
        trains a model using real-time data augmentation. for use with a small shapefile
        '''
        # generate data from which to upsample
        chip_gen = getIterData(train_shapefile, batch_size=chips_to_yield,
                               min_chip_hw=self.min_chip_hw, max_chip_hw=self.max_chip_hw,
                               classes=self.classes)
        X, Y = chip_gen.next()

        # image augmentaion via rotations
        datagen = ImageDataGenerator(rotation_range=180)
        datagen.fit(X)
        checkpointer = ModelCheckpoint(filepath="./models/ch_{epoch:02d}-{loss:.2f}.h5",
                                       verbose=1)

        if validation_prop:
            valX, valY = self._get_val_data(train_shapefile,
                                            int(validation_prop * train_size))
            self.model.fit_generator(datagen.flow(X,Y), samples_per_epoch=train_size,
                                     nb_epoch=nb_epoch, validation_data=(valX, valY),
                                     callbacks=[checkpointer])

        else:
            self.model.fit_generator(datagen.flow(X,Y), samples_per_epoch=train_size,
                                     nb_epoch=nb_epoch, callbacks=[checkpointer])

        if save_model:
            self.save_model(save_model)


    def retrain_output(self, X_train, Y_train, **kwargs):
        '''
        Retrains last dense layer of model. For use with unbalanced classes after
        training on balanced data.
        INPUT   (1) array 'X_train': training chips in the shape (train_size, 3, h, w)
                (2) list 'Y_train': one-hot associated labels to X_train. shape =
                (train_size, n_classes)
                (3) float 'validation_split': proportion of X_train to validate on.
                (4) string 'save_model': name of model for saving. if None, does not
                save model.
                (5) int 'nb_epoch': Number of epochs to train for
        OUTPUT  (1) retrained model
        '''
        # freeze all layers except final dense
        for i in xrange(len(self.model.layers[:-1])):
            self.model.layers[i].trainable = False

        # recompile model
        sgd = SGD(lr=self.lr_2, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # train model
        self.fit_xy(X_train, Y_train, **kwargs)

    def retrain_output_on_generator(self, train_shapefile, retrain_size=5000,
                                    save_model=None, nb_epoch=5, validation_prop=0.1):
        '''
        Retrains last dense layer of model with a generator. For use with unbalanced
        classes after training on balanced data.
        INPUT   (1) string 'train_shapefile': filename for the training data (must be a
                    geojson)
                (2) int 'train_size': number of chips to train model on. Defaults to 5000
                (5) string 'save_model': name of model for saving. if None, does not
                    save model.
                (6) int 'nb_epoch': Number of epochs to train for
        OUTPUT  (1) retrained model.
        '''
        # freeze all layers except final dense
        for i in xrange(len(self.model.layers[:-1])):
            self.model.layers[i].trainable = False

        # recompile model
        sgd = SGD(lr=self.lr_2, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # train model with frozen weights
        checkpointer = ModelCheckpoint(filepath="./models/ch_{epoch:02d}-{loss:.2f}.h5",
                                       verbose=1)

        data_gen = getIterData(train_shapefile, batch_size=self.batch_size,
                               min_chip_hw=self.min_chip_hw, max_chip_hw=self.max_chip_hw,
                               classes=self.classes)

        if validation_prop:
            valX, valY = self._get_val_data(train_shapefile,
                                            int(validation_prop * retrain_size))

            self.model.fit_generator(data_gen, samples_per_epoch=retrain_size,
                                     nb_epoch=nb_epoch, callbacks=[checkpointer],
                                     validation_data=(valX, valY))

        else:
            self.model.fit_generator(data_gen, samples_per_epoch=retrain_size)

        if save_model:
            self.save_model(save_model)


    def retrain_on_errors(self, X_train, Y_train, initial_weights, nb_epoch=5,
                        samples_per_epoch=2500, **kwargs):
        '''
        Retrain model on polygons that were initially misclassified.
        INPUT   (1) array 'X_train': misclassified chips. It is recommended to
                    only use chips with classification certainty under 0.9, given that
                    the training data is flawed. A shapefile of these chips can be
                    created  from filter_by_classification. use shape
                    (train_size, 3, h, w).
                (2) list 'Y_train': one-hot associated labels to X_train. shape =
                    train_size, n_classes)
                (3) int 'nb_epochs': number of epochs to train for.
                (4) string 'initial_weights': file path to weights to retrain on. It is
                    recommended to use weights from the first (balanced) round of
                    training, then repeat training on unbalanced after this.
                (5) int 'samples_per_epoch': number of samples to train on per epoch.
                    if this is more than len(X_train) real-time data augmentation will be
                    used
                (6) float 'validation_split': proportion of X_train to validate on.
                (7) string 'save_model': name of model for saving. if None, does not
                    save model.
        OUTPUT  (1) trained model
        '''
        # Recompile model to ensure all layers trainable
        for i in xrange(len(self.model.layers)):
            self.model.layers[i].trainable = True

        # Fit as usual if augmentation is unncessary
        if len(X_train) >= samples_per_epoch:
            self.fit_xy(X_train[:samples_per_epoch], Y_train[:samples_per_epoch],
                        nb_epoch=nb_epoch, **kwargs)

        else:
            # Augment data, fit on generator
            datagen = ImageDataGenerator(rotation_range=120, width_shift_range=0.2,
                                        height_shift_range=0.2, fill_mode='constant',
                                        cval=0, horizontal_flip= True, vertical_flip=True)

            # Fit model on batches with real-time data augmentation
            self.model.load_weights(initial_weights)
            self.model.fit_generator(datagen.flow(X_train, Y_train,
                                    batch_size=self.batch_size),
                                    samples_per_epoch=samples_per_epoch,
                                    nb_epoch=nb_epoch)

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

    def classify_shapefile(self, shapefile, output_name, img_name = None):
        '''
        Use the current model and weights to classify all polygons (of appropriate size)
        in the given shapefile. Records PoolNet classification, whether or not it was
        misclassified by PoolNet, and the certainty for the given class.
        INPUT   (1) string 'shapefile': name of the shapefile to classify
                (2) string 'output_name': name to give the classified shapefile
                (3) string 'image_name': name of the associated geotiff image if different
                    than catalog number. Defaults to None
        OUTPUT  (1) classified shapefile
        '''
        yprob, ytrue = [], []
        if output_name[-8:] != '.geojson':
            output_file = '{}.geojson'.format(output_name)
        else:
            output_file = output_name

        # Classify all chips in input shapefile
        print 'Classifying test data...'
        for x in get_iter_data(shapefile, batch_size = 5000, classes = self.classes,
                                  max_chip_hw=self.input_shape[1], img_name=img_name,
                                  min_chip_hw = self.min_chip_hw, return_labels=False):
            print 'Classifying polygons...'
            yprob += list(self.model.predict_proba(x)) # use model to predict classes

        # Get predicted class and certainty of classification results
        yhat = [np.argmax(i) for i in yprob]
        ycert = [np.max(j) for j in yprob]

        # Update shapefile, save as output_name
        data = zip(yhat, ycert)
        property_names = ['CNN_class', 'certainty']
        write_properties_to(data, property_names, shapefile, output_file)


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
            if cert >= min_cert and cert >= max_cert:
                ok_polygons.append(geom)

    # Save filtered polygons to nex shapefile
    filtrate = {data.keys()[0]: data.values()[0], data.keys()[1]: ok_polygons}
    with open('{}.geojson'.format(output_name), 'wb') as f:
        geojson.dump(filtrate, f)

    print 'Saved {} polygons as {}.geojson'.format(len(ok_polygons), output_name)
