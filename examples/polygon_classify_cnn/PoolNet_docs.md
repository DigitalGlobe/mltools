## PoolNet Docs  

## Table of Contents  
1. [About](#about)
2. [Methods](#methods)

### mltools.pool_net.PoolNet  
<i> class </i> mltools.pool_net. **PoolNet** ( <i> nb_classes=2, batch_size=32, min_chip_hw = 0, max_chip_hw = 125, input_shape=(3,125,125), n_dense_nodes=2048, fc=False, vgg=True, load_model=False, model_name=None, train_size=10000 </i> ) [[source](github.com/poolnet_class)]  
__________________________________________________________________________________________
#### About  
A convolutional neural network for pool detection in polygons.  

This classifier can be trained on polygons to detect pools in the property. This model uses [stochastic gradient descent](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/) to optimize [categorical crossentropy](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function).  

| **Parameter:**  | Description:                                                     |
|-----------------|------------------------------------------------------------------|
| classes | list['str'], The names of classes on which to train, exactly as they appear in the shapefile |
| batch_size | int, Number of chips per batch. Defaults to 32. |
| min_chip_hw | int, minimum acceptable side-dimension shape for each polygon. |
| max_chip_hw | int, maximum acceptable side-dimension shape for each polygon. |
| input_shape | tuple(ints), Shape of three-dim input image. Defaults to (3,125,125). Use Theano dimensional ordering. |
| fc   | bool, Use a fully convolutional model, instead of classic convolutional model. Defaults to False. |
| old_model | bool, Use a saved trained model (model_name) architecture and weights. Defaults to False. |
| model_name | string, Only relevant if old_model is True. Name of model (not including file extension) to load. Defaults to None (will not load a model) |
| train_size | int, Number of samples to train on per epoch if training on a generator. Defaults to 10000. |
| lr_1       | float, Learning rate to use on first round of training. Defaults to 0.001. |
| lr_2 | float, Learning rate to use on the second round of training (if applicable). Defaults to 0.01 |
__________________________________________________________________________________________
#### Methods  

1. [**make_fc_model**](#make_fc_model): Create a fully convolutional version of the current model.  
2. [**fit_xy**](#fit_xy): Train PoolNet on X and Y.  
3. [**fit_generator**](#fit_generator): For training PoolNet on more chips than will fit into memory.  
4. [**retrain_output**](#retrain_output): Retrain only the final layer of the model. For use on unbalanced data.  
5. [**save_model**](#save_model): Save the model architecture as a json and weights from final epoch.  
6. [**load_model**](#load_model): Use a previously saved model architecture.  
7. [**evaluate_model**](#evaluate_model): Get predicted classes and a classification report from test data.  
8. [**classify_shapefile**](#classify_shapefile): create a shapefile with classification results stored as properties.  

<i>**\__init__**(nb_classes=2, batch_size=32, input_shape=(3, 125, 125), fc = False, vgg=True, load_model=False, model_name=None, train_size=10000) </i>

##### make_fc_model  
Re-compile current model as [fully convolutional](https://arxiv.org/pdf/1411.4038.pdf). Beheads the standard convolutional model and adds three 2D convolutional layers.   

|Input| Description |
|---------------|------|
|*None* | *N/A*  |
|**Output** |  **Description** |
| model | Fully convolutional model ready for training|  


##### fit_xy  
\(X_train, Y_train, validation_split=0.1, save_model=None, nb_epoch=15)  
Fit the network on chips (X_train) and associated labels(Y_train). This can only be done if X and Y fit into memory.  

|Input| Description |
|---------------|------|
|X_train | array, training chips with shape *(n_chips, n_channels, chip_h, chip_w)*|
|Y_train | array, training chip labels using one-hot encoding |
|validation_split | float, roportion of X_train to use as validation data |
|save_model | string, name under which to save model. Defaults to None (doesn't save model) |
|nb_epoch | int, number of epochs to train for |
|**Output** |  **Description** |
|trained model | model trained on X_train |  


##### fit_generator  
\(train_shapefile, gen_batch_size=1000, batches_per_epoch=2, validation_split=0.1, save_model=None, nb_epoch=5)  
 Train PoolNet on the mltools data generator ([data_extractors.get_iter_data](https://github.com/kostasthebarbarian/mltools/blob/master/mltools/data_extractors.py)). Note that the train size will be gen_batch_size * batches_per_epoch, not self.train_size.  

|Input| Description |
|---------------|------|
|train_shapefile | string, filepath to shapefile containing polygons to train model on|
|gen_batch_size | int, number of chips to generate per batch of training. This must fit in memory. |
|batches_per_epoch | int, number of batches to generate and train on per epoch. Total number of chips trained on = *batches x batches_per_epoch* |
|validation_split | float, proportion of training data to use for validation |
|save_model | string, name under which to save model. Defaults to None (doesn't save model) |
|nb_epoch | Number of epochs to train for |
|**Output** |  **Description** |
|trained model | model trained on polygons in shapefile |  


##### retrain_output  
\(X_train, Y_train, kwargs)  
 Re-train the final dense layer of PoolNet. This is meant for use on unbalanced classes, in order to minimize false positives associated with the initial training on balanced classes.  

|Input| Description |
|---------------|------|
|X_train | array, training chips with shape *(n_chips, n_channels, chip_h, chip_w)*|
|Y_train | array, training chip labels using one-hot encoding |
|kwargs | Keyword arguments from fit_xy |
|**Output** |  **Description** |
|trained model | Model with last dense layer trained on X_train |

##### retrain_output_on_generator  
\(train_shapefile, gen_batch_size=2500, batches_per_epoch=2, validation_split=0.1, save_model=None, nb_epoch=5)  
 Re-train the final dense layer of PoolNet on mltools generator. This is meant for use on unbalanced classes, in order to minimize false positives associated with the initial training on balanced classes. Use generator when train size is too large to fit into memory.

 |Input| Description |
 |---------------|------|
 |train_shapefile | string, filepath to shapefile containing polygons to train model on|
 |gen_batch_size | int, number of chips to generate per batch of training. This must fit in memory. |
 |batches_per_epoch | int, number of batches to generate and train on per epoch. Total number of chips trained on = *batches x batches_per_epoch* |
 |validation_split | float, proportion of training data to use for validation |
 |save_model | string, name under which to save model. Defaults to None (doesn't save model) |
 |nb_epoch | Number of epochs to train for |
 |**Output** |  **Description** |
 |retrained model | model retrained on unbalanced classes as found in shapefile |


##### save_model  
\(model_name)  
Saves model architecture as json and weights as h5py doc.  

|Input| Description |
|---------------|------|
|model_name | string, name under which to save model architecture and weights|
|**Output** |  **Description** |
|model_name.json | model architeture |
|model_name.h5 | model weights |  


##### load_model  
\(model_name)  
Load model architecture. If you wish to load weights as well you must call self.model.load_weights('weight_file.h5') after calling this function.  

|Input| Description |
|---------------|------|
|model_name | string, filepath and name under which model architecture is saved, |
|**Output** |  **Description** |
|model | model loaded with input architecture. |  


##### evaluate_model  
\(X_test, Y_test, return_yhat)  
Classify X_test chips and print a classification report from the trained model.  

|Input| Description |
|---------------|------|
|X_test | array, test chips to classify |
|Y_test | array, labels for X_test with [one-hot encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) |
|return y_hat | bool, True to return predicted y values |
|**Output** |  **Description** |
|yhat | array, predicted classes for X_test |  


##### classify_shapefile  
\(shapefile, output_name, img_name = None)  
Create a geojson with results of classification saved as properties for each polygon.  

|Input| Description |
|---------------|------|
|shapefile | string, name of shapefile to classify. Will automatically filter out polygons that are too large (side dimensions larger than input_shape[1]) |
|output_name | string, name under which to save the output file |
|img_name | name of the associated geotiff image if different than catalog number. Defaults to None. |  
|**Output** |  **Description** |
|output_name | file with polygons with classification results as properties |
