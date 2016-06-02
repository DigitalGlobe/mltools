import numpy as np
import random
import json
from mltools.data_extractors import get_iter_data
from mltools.geojson_tools import write_properties_to
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation, Flatten, Reshape
from keras.models import Sequential, Graph, model_from_json
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
    Fully Convolutional model to classify polygons as pool/no pool

    INPUT   (1) int 'nb_classes': number of different image classes. defaults to 2
            (pool/no pool)
            (2) int 'batch_size': amount of images to train for each batch. defaults
            to 32
            (3) tuple[int] 'input_shape': shape of input images (3-dims). defaults to
            (3,125,125)
            (4) bool 'fc': True for fully convolutional model, else classic convolutional
            model. defaults to False.
            (5) bool 'vgg': True to use vggnet architecture. Defaults to True (currently
            better than original)
            (6) bool 'load_model': Use a saved trained model (model_name) architecture
            and weights. Defaults to False
            (7) string 'model_name': Only relevant if load_model is True. name of model
            (not including file extension) to load. Defaults to None
            (8) int 'train_size': number of samples to train on per epoch. defaults to
            10000
    '''

    def __init__(self, nb_classes=2, batch_size=32,
                input_shape=(3, 125, 125), fc = False,
                vgg=True, load_model=False, model_name=None, train_size=10000):

        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.fc = fc
        self.vgg = vgg
        self.load_model = load_model
        self.train_size = train_size
        if self.vgg:
            self.model = self._VGG_16()
        elif self.load_model:
            self.model_name = model_name
            self.model = self.load_model_weights(model_name)
        else:
            self.model = self._compile_model()
        self.model_layer_names = [self.model.layers[i].get_config()['name']
                                    for i in range(len(self.model.layers))]
        if self.fc:
            self.model = self.make_fc_model()

    def _compile_model(self):
        '''
        compiles standard convolutional netowrk (not FCNN)
        currently not a good model. use VGGnet.
        '''
        print 'Compiling standard model...'
        model = Sequential()

        model.add(Convolution2D(64, 5, 5, W_regularizer = l1l2(l1=0.01, l2=0.01),
                                border_mode = 'valid',
                                input_shape=self.input_shape,
                                activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        model.add(Dropout(0.75))

        model.add(Convolution2D(128, 3, 3, W_regularizer = l1l2(l1=0.01, l2=0.01),
                                border_mode = 'valid',
                                activation = 'relu'))
        model.add(BatchNormalization(mode=0, axis=1))
        model.add(MaxPooling2D(pool_size = (2,2)))

        model.add(Convolution2D(128, 3, 3, W_regularizer = l1l2(l1=0.01, l2=0.01),
                                border_mode = 'valid',
                                activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, 3, 3, W_regularizer = l1l2(l1=0.01, l2=0.01),
                                border_mode = 'valid',
                                activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides=(1,1)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, 3, 3,
                                border_mode = 'valid',
                                activation = 'relu'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.0001, decay=0.01)

        model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')

        return model


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
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9, nesterov=True)
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
        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return model


    def fit_xy(self, X_train, Y_train, nb_epoch=15, validation_split=0.1,
               save_model = None):
        '''
        Fit model on pre-loaded training data. Only for sizes small enough to fit in
        memory (~ 10000 3x100x100 chips on dg_gpu)
        INPUT   (1) array 'X_train': training chips in the shape (train_size, 3, h, w)
                (2) list 'Y_train': one-hot associated labels to X_train. shape =
                train_size, n_classes)
                (3) float 'validation_split': proportion of X_train to validate on.
                #TODO: add X_test and Y_test, set val_split to None
                (4) string 'save_model': name of model for saving. if None, does not
                save model.
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


    def fit_generator(self, train_shapefile, batches = 10000, batches_per_epoch=5,
                      min_chip_hw=30, max_chip_hw=125, validation_split=0.1,
                      save_model=None, nb_epoch=5):
        '''
        Fit a model using a generator that yields a large batch of chips to train on.
        INPUT   (1) string 'train_shapefile': filename for the training data (must be a
                geojson)
                (2) int 'batches': number of chips to yield. must be small enough to fit
                into memory.
                (3) int 'min_chip_hw': minimum acceptable side dimension for polygons
                (4) int 'max_chip_hw': maximum acceptable side dimension for polygons
                (5) float 'validation_split': proportion of chips to use as validation
                data.
                (6) string 'save_model': name of model for saving. if None, does not
                save model.
        OUTPUT  (1) trained model.
        '''
        es = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
        checkpointer = ModelCheckpoint(filepath="./models/ch_{epoch:02d}-{val_loss:.2f}.h5",
                                       verbose=1)
        ct = 0

        # iterate through batches, train model on each
        for e in range(nb_epoch):
            print 'Epoch {}/{}'.format(e + 1, nb_epoch)
            for X_train, Y_train in get_iter_data(train_shapefile,
                                                  batch_size = batches,
                                                  min_chip_hw = min_chip_hw,
                                                  max_chip_hw = max_chip_hw,
                                                  resize_dim = self.input_shape):
                # Train on batch
                self.model.fit(X_train, Y_train, batch_size=32, nb_epoch=1,
                               validation_split=validation_split,
                               callbacks=[checkpointer])

                # Go to next epoch if batches_per_epoch have been trained
                ct += 1
                if ct == batches_per_epoch:
                    break

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
                #TODO: add X_test and Y_test, set val_split to None
                (4) string 'save_model': name of model for saving. if None, does not
                save model.
        OUTPUT  (1) retrained model
        '''
        # freeze all layers except final dense
        for i in xrange(len(self.model.layers[:-1])):
            self.model.layers[i].trainable = False

        # recompile model
        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # train model
        self.fit_xy(X_train, Y_train, **kwargs)

    def save_model(self, model_name):
        '''
        INPUT string 'model_name': name to save model and weigths under, including
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

    def load_model_weights(self, model_name):
        '''
        INPUT  (1) string 'model_name': filepath to model and weights, not including
        extension. base name should be the same for both model and weights.
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in
        fit_model method
        '''
        print 'Loading model {}'.format(self.model_name)
        model = '{}.json'.format(self.model_name)
        weights = '{}.h5'.format(self.model_name)

        #load model
        with open(model) as f:
            m = f.next()
        mod = model_from_json(json.loads(m))

        #load weights
        mod.load_weights(weights)
        print 'Done.'
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

    def classify_shapefile(self, shapefile, output_name):
        yhat, ytrue = [], []

        # Classify all chips in input shapefile
        print 'Classifying test data...'
        for x, y in get_iter_data(shapefile, batch_size = 5000,
                                      max_chip_hw=self.input_shape[1]):
            print 'Classifying polygons...'
            yhat += list(self.model.predict_classes(x)) # use model to predict classes
            ytrue += [int(i[1]) for i in y] # put ytest in same format as ypred

        # find misclassfied chips
        missed = [0 if ytrue[i] == yhat[i] else 1 for i in xrange(len(yhat))]

        # Update shapefile, save as output_name
        data = zip(yhat, missed)
        property_names = ['PoolNet_class', 'missed']
        write_properties_to(data, property_names, shapefile, output_name)


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
