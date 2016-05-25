import numpy as np
import random
import json
from polygon_pipeline import get_iter_data
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.models import Sequential, Graph, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import classification_report

class PoolNet(object):
    '''
    Fully Convolutional model to classify polygons as pool/no pool

    INPUT   (1) int 'nb_chan': number of input channels. defaults to 3 (rgb)
            (2) int 'nb_epoch': number of epochs to train. defaults to 4
            (3) int 'nb_classes': number of different image classes. defaults to 2 (pool/no pool)
            (4) int 'batch_size': amount of images to train for each batch. defaults to 32
            (5) list[int] 'input_shape': shape of input images (3-dims). defaults to (3,224,224)
            (6) int 'n_dense_nodes': number of nodes to use in dense layers. defaults to 2048.
            (7) bool 'fc': True for fully convolutional model, else classic convolutional model. defaults to False.
            (8) bool 'vgg': True to use vggnet architecture. Defaults to True (currently better than original)
            (9) bool 'load_model': Use a saved trained model (model_name) architecture and weights. Defaults to False
            (10) string 'model_name': Only relevant if load_model is True. name of model (not including file extension) to load. Defaults to None
            (11) int 'train_size': number of samples to train on per epoch. defaults to 5000
    '''

    def __init__(self, nb_chan=3, nb_epoch=4, nb_classes=2, batch_size=32,
                input_shape=(3, 224, 224), n_dense_nodes = 2048, fc = False,
                vgg=False, load_model=False, model_name=None, train_size=500):

        self.nb_epoch = nb_epoch
        self.nb_chan = nb_chan
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_dense_nodes = n_dense_nodes
        self.fc = fc
        self.vgg = vgg
        self.load_model = load_model
        self.train_size = train_size
        if self.vgg:
            self.model = self.VGG_16()
        # elif self.alexnet:
        #     self.model = self.AlexNet()
        elif self.load_model:
            self.model_name = model_name
            self.model = self.load_model_weights()
        else:
            self.model = self.compile_model()
        self.model_layer_names = [self.model.layers[i].get_config()['name']
                                    for i in range(len(self.model.layers))]
        if self.fc:
            self.model = self.make_fc_model()

    def compile_model(self):
        '''
        compiles standard convolutional netowrk (not FCNN)
        '''
        print 'Compiling standard model...'
        model = Sequential()

        model.add(Convolution2D(96, 7, 7,
                                border_mode = 'valid',
                                input_shape=self.input_shape,
                                activation = 'relu'))
        model.add(Dropout(0.75))

        model.add(Convolution2D(128, 5, 5,
                                border_mode = 'valid',
                                activation = 'relu'))
        # model.add(BatchNormalization(mode=0, axis=-1))
        model.add(MaxPooling2D(pool_size = (3,3),
                                strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(256, 3, 3,
                                border_mode = 'valid',
                                activation = 'relu'))
        # model.add(BatchNormalization(mode=0, axis=-1))
        model.add(MaxPooling2D(pool_size = (3,3),
                                strides=(2,2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(256, 3, 3,
                                border_mode = 'valid',
                                activation = 'relu'))
        model.add(Dropout(0.5))

        model.add(Convolution2D(256, 3, 3,
                                border_mode = 'valid',
                                activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (3,3),
                                strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(self.n_dense_nodes))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_dense_nodes))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)

        model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')

        return model


    def VGG_16(self):
        '''
        Implementation of VGG 16-layer net. currently too large for memory on dg_gpu
        '''
        print 'Compiling VGG Net...'

        model = Sequential()
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=self.input_shape))
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

    def get_behead_index(self, layer_names):
        '''
        helper function to find index where net flattens
        INPUT   (1) list 'layer_names': names of each layer in model
        OUTPUT  (1) int 'behead_ix': index of flatten layer
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
        inp_shape = self.model.layers[behead_ix - 1].get_output_shape_at(-1) # shape of image entering FC layers

        # replace dense layers with convolutions
        model = Sequential()
        model_layers += [Convolution2D(self.n_dense_nodes, 1, 1)]
        model_layers += [Activation('relu')]
        model_layers += [Convolution2D(self.n_dense_nodes, 1, 1)]
        model_layers += [Activation('relu')]
        model_layers += [Convolution2D(self.nb_classes, inp_shape[-1], inp_shape[-1])]
        model_layers += [Reshape((self.nb_classes-1,1))] # must be same shape as target vector (None, num_classes, 1)
        model_layers += [Activation('softmax')]

        print 'Compiling Fully Convolutional Model...'
        for process in model_layers:
            model.add(process)
        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return model

    def train_on_data(self, train_shapefile, val_shapefile=None, min_chip_hw=100,
                      max_chip_hw=224, validation_split=0.15):
        '''
        Uses generator to train model from shapefile

        INPUT   (1) string 'train_shapefile': geojson file containing polygons to be trained on
                (2) string 'val_shapefile': geojson file containing polygons for validation. use a shuffled version of the original balanced shapefile
                (3) int 'min_chip_hw': minimum acceptable side dimension for polygons
                (4) int 'max_chip_hw': maximum acceptable side dimension for polygons
                (5) float 'validation_split': amount of sample to validate on relative to train size. set to zero to skip validation. defaults to 0.15
        OUTPUT  (1) trained model
        '''

        print 'Training model on batches...'

        # callback for early stopping
        es = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

        # create generators for train and validation data
        data_gen = get_iter_data(train_shapefile,
                                batch_size=self.batch_size,
                                min_chip_hw=min_chip_hw,
                                max_chip_hw=max_chip_hw,
                                resize_dim=self.input_shape)

        if val_shapefile:
            val_gen = get_iter_data(val_shapefile,
                                    batch_size=self.batch_size,
                                    min_chip_hw=min_chip_hw,
                                    max_chip_hw=max_chip_hw,
                                    resize_dim=self.input_shape)

            # fit model
            self.model.fit_generator(data_gen,
                                    samples_per_epoch=self.train_size,
                                    nb_epoch=self.nb_epoch,
                                    callbacks=[es], validation_data=val_gen,
                                    nb_val_samples=int(self.train_size * validation_split))
        else:
            self.model.fit_generator(data_gen,
                                    samples_per_epoch=self.train_size,
                                    nb_epoch=self.nb_epoch, callbacks=[es])

    def retrain_output(self, train_shapefile, **kwargs):
        '''
        Retrains last dense layer of model. For use with unbalanced classes after
        training on balanced data.
        INPUT   (1) string 'train_shapefile': shapefile containing polygons to retrain model on
                (2) string 'val_shapefile': geojson file containing polygons for validation. use a shuffled version of the original balanced shapefile
                (3) int 'min_chip_hw': minimum acceptable side dimension for polygons
                (4) int 'max_chip_hw': maximum acceptable side dimension for polygons
                (5) float 'validation_split': amount of sample to validate on relative to train size. set to zero to skip validation. defaults to 0.15
        OUTPUT  (1) retrained model
        '''
        # freeze all layers except final dense
        for i in xrange(len(self.model.layers[:-2])):
            self.model.layers[i].trainable = False

        # recompile model
        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # train model
        self.train_on_data(train_shapefile, **kwargs)

    def save_model(self, model_name):
        '''
        INPUT string 'model_name': name to save model and weigths under, including
        filepath but not extension
        Saves current model as json and weigts as h5df file
        '''
        model = '{}.json'.format(model_name)
        weights = '{}.h5'.format(model_name)
        json_string = self.model.to_json()
        self.model.save_weights(weights)
        with open(model, 'w') as f:
            json.dump(json_string, f)

    def load_model_weights(self, model_name):
        '''
        INPUT  (1) string 'model_name': filepath to model and weights, not including
        extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in
        fit_model method
        '''
        print 'Loading model {}'.format(self.model_name)
        model = '{}.json'.format(self.model_name)
        weights = '{}.h5'.format(self.model_name)
        with open(model) as f:
            m = f.next()
        mod = model_from_json(json.loads(m))
        mod.load_weights(weights)
        print 'Done.'
        return mod

    def evaluate_model(self, X_test, y_test, return_yhat=False):
        '''
        Predicts classes of X_test and evaluates precision, recall and f1 score
        INPUT   (1) array 'X_test': array of chips
                (2) list 'y_test': labels corresponding to chips in X_test
                (3) bool 'return_yhat': return the values of predicted classes for X_test
        OUTPUT  (1) classification report
        '''
        y_hat = self.model.predict_classes(X_test)
        print classification_report(y_test, y_hat)

        if return_yhat:
            return y_hat

    def confusion_matrix_imgs(self, X_test, y_test, y_pred):
        '''
        Generate file with incorrectly classified polygons for inspection
        INPUT   (1) array 'X_test': array of chips
                (2) list 'y_test': labels corresponding to chips in X_test
                (3) list 'y_pred': results of classification on X_test. if None, will classify X_test and generate y_pred.
        OUTPUT  (1) true positives
                (2) true negatives
                (3) false positives
                (4) false negatives
        '''
        wrong = X_test[[y_test!=y_pred]]
        fp, fn = wrong[[y_test==0]], wrong[[y_test==1]]

        right = X_test[[y_test==y_pred]]
        tp, tn = right[[y_test==1]], right[[y_test==0]]

        return tp, tn, fp, fn


## GRAVEYARD ##
        # in PoolNet.train_on_data:
        # for epoch in xrange(self.nb_epoch):
        #
        #     print 'Epoch {}:'.format(epoch)
        #     for chips, labels in get_iter_data(shapefile, batch_size=self.batch_size, min_chip_hw=20, max_chip_hw=224, return_labels=True, mask=True):
        #         X = np.array([i[:3] for i in chips])
        #         y = [1 if label == 'Swimming pool' else 0 for label in labels]
        #         Y = np_utils.to_categorical(y, self.nb_classes)
        #
        #         print 'Training...'
        #         import pdb; pdb.set_trace()
        #         mod.train_on_batch(X, Y)



    # def AlexNet(self):
    #     '''
    #     Implementation of AlexNet
    #     '''
    #     print 'Compiling AlexNet...'
    #
    #     inputs = Input(shape=(3,224,224))
    #
    #     conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
    #                            name='conv_1')(inputs)
    #
    #     conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    #     conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    #     conv_2 = ZeroPadding2D((2,2))(conv_2)
    #     conv_2 = merge([Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(splittensor(ratio_split=2,id_split=i)(conv_2)) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")
    #
    #     conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    #     conv_3 = crosschannelnormalization()(conv_3)
    #     conv_3 = ZeroPadding2D((1,1))(conv_3)
    #     conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)
    #
    #     conv_4 = ZeroPadding2D((1,1))(conv_3)
    #     conv_4 = merge([
    #         Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
    #             splittensor(ratio_split=2,id_split=i)(conv_4)
    #         ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")
    #
    #     conv_5 = ZeroPadding2D((1,1))(conv_4)
    #     conv_5 = merge([
    #         Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
    #             splittensor(ratio_split=2,id_split=i)(conv_5)
    #         ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")
    #
    #     dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)
    #
    #     dense_1 = Flatten(name="flatten")(dense_1)
    #     dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    #     dense_2 = Dropout(0.5)(dense_1)
    #     dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    #     dense_3 = Dropout(0.5)(dense_2)
    #     dense_3 = Dense(2,name='dense_3')(dense_3)
    #     prediction = Activation("softmax",name="softmax")(dense_3)
    #
    #     model = Model(input=inputs, output=prediction)
    #     sgd = SGD(lr=0.001, decay=0.01, momentum=0.9, nesterov=True)
    #     model.compile(loss='categorical_crossentropy', optimizer='sgd')
    #
    #     return model
