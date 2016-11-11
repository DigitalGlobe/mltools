## PoolNet Docs  

## Table of Contents  
1. [About](#about)
2. [Methods](#methods)

### mltools.pool_net.PoolNet  
<i> class </i> mltools.pool_net. **PoolNet** ( <i> classes, batch_size=32, input_shape=(3,125,125), small_model=False, model_name=None, learning_rate=0.001, kernel_size=3 </i> ) [[source](github.com/poolnet_class)]  
__________________________________________________________________________________________
#### About  
A Convolutional Neural Network (CNN) for classification of image chips.  

This class creates a generic classifier by training on labeled image chips. The chips can be stored in a geojson with associated image strips in the  dame directory, or on preloaded chips. The class can also deploy the model on target data in a geojson. The model uses [stochastic gradient descent](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/) to optimize [categorical crossentropy](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function).  

| **Parameter:**  | Description:                                                     |
|-----------------|------------------------------------------------------------------|
| classes | list['str'], The names of classes on which to train, exactly as they appear in the geojson with training data |
| batch_size | int, Number of chips per batch. Defaults to 32. |
| input_shape | tuple(ints), Shape of three-dimensional input image. Defaults to (3,125,125). Use Theano dimensional ordering (bands, rows, cols). |
| small_model | bool, Use a smaller model (9 layers instead of 16). This will speed up training but may not fit into memory for larget inputs. Defaults to False. |
| model_name | string, Name of a model in the working directory (not including file extension) to load. The architecture should be saved as model_name.json and the weights as model_weights.h5. Use this if you already have a saved model and simply want to deploy on target data. Defaults to None (will not load a model) |
| learning_rate | float, Learning rate to use on the first round of training. Defaults to 0.001. |
| kernel_size | int, Size of the convolutional kernels to use at each layer of the model. Defaults to 3 (standard for VGGNet). |
__________________________________________________________________________________________
#### Methods  

1. [**save_model**](#save_model): Save the model architecture as a json and current weights as an h5 file.  
2. [**fit_from_geojson**](#fit_from_geojson): Fit the model using training data stored in a geojson file.  
3. [**fit_xy**](#fit_xy): Fit a model using chips and labels already loaded into memory.  
4. [**classify_geojson**](#classify_geojson): Deploy the model on all chips in a target geojson file. Produces a classified version of the input geojson.

<i>**\__init__**(classes=['No swimming pool', 'Swimming pool'], batch_size=32, input_shape=(3, 125, 125), small_model = False, model_name=None, learning_rate=0.001, kernel_size=3) </i>

##### save_model  
\(model_name)  

Saves model architecture as json and weights as h5py doc.  

|Input| Description |
|---------------|------|
|model_name | string, name under which to save model architecture and weights|
|**Output** |  **Description** |
|model_name.json | model architeture |
|model_name.h5 | model weights |   


##### fit_from_geojson

\(train_geojson, max_side_dim=None, min_side_dim=0, chips_per_batch=5000, train_size=10000, validation_split=0.1, bit_depth=8, save_model=None, nb_epoch=10, shuffle_btwn_epochs=True, return_history=False, save_all_weights=True, retrain=False, learning_rate_2=0.01)  


Fit a model from a geojson file with training data. This method iteratively
    yields large batches of chips to train on for each epoch. Please ensure that
    your current working directory contains all imagery referenced in the
    image_id property in train_geojson, and are named as follows: <image_id>.tif,
    where image_id is the catalog id of the image.

|Input| Description |
|---------------|------|
|train_geojson | string, Geojson file containing training data. This file must be filtered such that all polygons are of valid size (as defined by max_side_dim and min_side_dim) |
|max_side_dim | int, Maximum acceptable side dimension (in pixels) for a chip. If None, defaults to input_shape[-1]. If larger than the input shape the chips extracted will be downsampled to match the input shape. Defaults to None. |
|min_side_dim | int, Minimum acceptable side dimension (in pixels) for a chip. Defaults to 0. |
|chips_per_batch | Number of chips to yield per batch. Must be small enough to fit into memory. Defaults to 5000 (decrease for max_side_dim > 125). |
|train_size | int, Number of chips to use for training data. |
|validation_split | float, Proportion of training chips to use as validation data. Defaults to 0.1. |
|bit_depth | int, Bit depth of the image strips from which training chips are extracted. Defaults to 8 (standard for DRA'ed imagery).|
|save_model | string, Name of model for saving. if None, does not save model to disk. Defaults to None |
|nb_epoch | int, Number of epochs to train for. Each epoch will be trained on batches * batches_per_epoch chips. Defaults to 10.|
|shuffle_btwn_epochs | bool, Shuffle the features in train_geojson between each epoch. Defaults to True. |
|return_history | bool, Return a list containing metrics from past epochs. Defaults to False. |
|save_all_weights | bool, Save model weights after each epoch. A directory called models will be created in the working directory. Defaults to True. |
|retrain | bool, Freeze all layers except final softmax to retrain only the final weights of the model. Defaults to False. |
|learning_rate_2 | float, Learning rate for the second round of training. Only relevant if retrain is True.|
|**Output** |  **Description** |
|history | dict, A full history of validation loss after each epoch. |

##### fit_xy  
\(X_train, Y_train, validation_split=0.1, save_model=None, nb_epoch=10, shuffle_btwn_epochs=True, return_history=False, save_all_weights=True, retrain=False, learning_rate_2=0.01)  

Fit the network on chips (X_train) and associated labels(Y_train).  

|Input| Description |
|---------------|------|
|X_train | array, Training chips with the following dimensions: (train_size, num_channels, rows, cols). Dimensions of each chip should match the input_size to the model.|
|Y_train | array, One-hot encoded labels to X_train with dimensions as follows: (train_size, n_classes) |
|validation_split | float, Proportion of X_train to validate on while training. |
|save_model | string, Name under which to save model. if None, does not save model. Defualts to None. |
|nb_epoch | int, Number of training epochs to complete |
|shuffle_btwn_epochs | bool, Shuffle the features in train_geojson between each epoch. Defaults to True. |
|return_history | bool, Return a list containing metrics from past epochs. Defaults to False. |
|save_all_weights | bool, Save model weights after each epoch. A directory called models will be created in the working directory. Defaults to True. |
|retrain | bool, Freeze all layers except final softmax to retrain only the final weights of the model. Defaults to False. |
|learning_rate_2 | float, Learning rate for the second round of training. Only relevant if retrain is True.|
|**Output** |  **Description** |
|history | dict, A full history of validation loss after each epoch. |


##### classify_geojson
\(target_geojson, output_name, max_side_dim=None, min_side_dim=0, numerical_classes=True, chips_in_mem=5000, bit_depth=8)  

Use the current model and weights to classify all polygons in target_geojson. The output file will have a 'CNN_class' property with the net's classification result, and a 'certainty' property with the net's certainty in the assigned classification. Please ensure that your current working directory contains all imagery referenced in the image_id property in target_geojson, and are named as follows: <image_id>.tif, where image_id is the catalog id of the image.

|Input| Description |
|---------------|------|
|target_geojson | string, Name of the geojson to classify. This file should only contain chips with side dimensions between min_side_dim and max_side_dim (see below). |
|output_name | string, Name under which to save the classified geojson. |
|max_side_dim | int, Maximum acceptable side dimension (in pixels) for a chip. If None, defaults to input_shape[-1]. If larger than the input shape the chips extracted will be downsampled to match the input shape. Defaults to None.|
|min_side_dim | int, Minimum acceptable side dimension (in pixels) for a chip. Defaults to 0.|
|numerical_classes | bool, Make output classifications correspond to the indicies (base 0) of the 'classes' attribute. If False, 'CNN_class' is a string with the class name. Defaults to True. |
|chips_in_mem | int, Number of chips to load in memory at once. Decrease this parameter for larger chip sizes. Defaults to 5000. |
|bit_depth | int, Bit depth of the image strips from which training chips are extracted. Defaults to 8 (standard for DRA'ed imagery). |
