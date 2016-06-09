# Pool Detection Using Deep Learning

## Table of Contents

1. [About PoolNet](#about-poolnet)
    * [The Challenge](#the-challenge)
    * [Network Architecture](#network-architecture)
2. [Getting Started](#getting-started)
    * [Setting up your EC2 Instance](#setting-up-an-ec2-instance-with-theano)
    * [Setting up a Virtual Environment](#setting-up-an-environment)
3. [PoolNet Workflow](#poolnet-workflow)
4. [Data Selection and Training](#data-selection-and-training)
    * [Image Preprocessing](#image-preprocessing)
    * [Two-Phase Training](#two-phase-training)
    * [Misclassified Polygons](#misclassified-polygons)
5. [Results](#results)
    * [Original Data](#original-data)
    * [Ground Truth](#ground-truth)
6. [Docs](#docs)

## About PoolNet

PoolNet uses deep learning with a convolutional neural network to classify satellite images of various property polygons as homes with or without a pool. This model provides an efficient and reliable way to classify polygons, information that is valuable to insurance companies and would otherwise be challenging to collect at a large scale. With appropriate training data this model can be extended to various applications such as vehicles, boats, solar panels and buildings.

<img alt='Example property polygons. Red indicates no pool, green indicates that there is a pool within the polygon.' src='images/sample_polygons.png' width=400>  
<sub> Example property polygons. Red indicates no pool, green indicates that there is a pool within the polygon. </sub>  

### The Challenge

Pools turn out to be very diverse in satellite images, varying in shape, color, tree-coverage and location. A convolutional neural net is therefore a promising option for machine detection of pools, providing the flexibility of learning common abstract qualities of the item of interest independent of location in the input image. The vast number of parameters trained in PoolNet allows it to learn a variety of features that pools have that other machine learning techniques may overlook.

<img alt='Major challenge in machine classification of pools- diversity of pools in satellite imagery' src='images/pool_diversity.png'>  
<sub>Various pool-containing polygons from the test data. Notice the diversity in shape, size, color, intensity and location in the polygon. This makes machine classification of pools very challenging. </sub>  


### Network Architecture

PoolNet utilizes the [VGG-16](https://arxiv.org/pdf/1409.1556.pdf) network architecture, a 16-layer convolutional neural network, the top-scoring submission for the 2014 [ImageNet](http://www.image-net.org/challenges/LSVRC/2014/) classification challenge. This architecture is composed of many small (3x3) convolutional filters, which enables such a deep network to be trained and deployed in a timely manner on a GPU.  

<b>VGG-16 Architecture</b>  
<img src='images/VGGNet.png' width=500>  
<sub>Architecture of VGGNet. The input layer (blue) is comprised of a 3-channel pansharpened rgb property polygon. Each green layer represents a convolution of the original image with a 3x3 kernel. Max-pooling layers (denoted by the black MP) are used for downsampling the convolved image by taking only the most intense of a given pool of pixels. The yellow layers at the end represent fully connected layers where all remaining pixels are flattened into a 2-dimensional layer of nodes. Finally two nodes of the the softmax layer at the end each represent one class (pool or no pool) and produces a probability the the input image belongs to that class. </sub>  

## Getting Started  

PoolNet requires a GPU to prevent training from being prohibitively slow. Before getting started you will need to set up an EC2 instance with Theano.

### Setting up an EC2 Instance With Theano  

Begin by setting up an Ubuntu g2.2xlarge EC2 GPU ubuntu instance on AWS.  

Follow steps 1 - 9 on [this tutorial](http://markus.com/install-theano-on-aws/).  
In short:

1. Update packages:  

        sudo apt-get update  
        sudo apt-get -y dist-upgrade

2. Open tmux:  

        tmux  

3. Install dependencies:  

        sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy

4. Install bleeding-edge version of Theano:  

        sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git  

5. Get cuda toolkit (7.0):  

        sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb  
6. Depackage cuda:  

        sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb  

7. Add package and install cuda driver (~5 min)  

        sudo apt-get update  
        sudo apt-get install -y cuda  

8. Add cuda nvcc and ld_library_path to path:  

        echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc  

9. Reboot:  

        sudo reboot  

10. Create a .theanorc in the /home/ubuntu/ directory as follows:  

<b>.theanorc config: </b>  

        [global]  
        floatX = float32  
        device = gpu  
        optimizer = fast_run  

        [lib]  
        cnmem = 0.9

        [nvcc]  
        fastmath = True

        [blas]  
        ldflags = -llapack -lblas  


### Setting up an environment

1. Create the environment:  

        conda create -n geo python ipython numpy scipy gdal git libgdal=2  

2. upgrade pip:  

        pip install --upgrade pip  

3. install mltools:  

        pip install mltools

## PoolNet Workflow


Start with a geojson shapefile ('shapefile.geojson') and associated tif images:  

<img alt='Raw shapefile polygons overlayed on tif' src='images/raw_polygons.png' width=750>   
<sub> Pansharpened tif image with associated polygons overlayed. Green polygons indicate there is a pool in the property. </sub>

### Prepare Shapefile for Training

1. Filter shapefile for legitimate polygons. Use resolution to determine minimum and maximum acceptable chip side dimensions (generally between 30 and 125 pixels for pansharpened images).  

    <img alt='Small polygons to be filtered out of shapefile' src='images/small_polygons.png' height=200>
    <img alt='Shapefile with small polygons filtered out' src='images/filtered_polygons.png' height=200>

        import mltools.geojson_tools as gt
        import mltools.data_extractors as de

        gt.filter_polygon_size('shapefile.geojson', 'filtered_shapefile', min_polygon_hw=30, max_polygon_hw=125)
        # creates filtered_shapefile.geojson

2. Create train and test shapefiles with and without balanced classes.

        gt.create_balanced_geojson('filtered_shapefile.geojson', output_name = 'filtered', 'balanced = False', train_test = 0.2)
        # creates train_filtered.geojson and test_filtered.geojson

        # use train_filtered.geojson to create balanced class shapefile:
        gt.create_balanced_geojson('train_filtered.geojson', output_name = 'train_balanced')
        # creates training data (train_balanced.geojson) with balanced classes for first round of training  

    <b>Notice that we now have the following shapefiles</b>:  

    <img alt='Schema for shapefiles created from the original raw data.' src='images/repr_shapefiles.png' width=200>  

    a. <b>shapefile.geojson</b>: original file with all polygons  
    b. <b>filtered_shapefile.geojson</b>: file with all polygons with side dimensions between 30 and 125 pixels  
    c. <b>test_filtered.geojson</b>: test data with filtered polygons and unbalanced classes. don't touch it until testing the model!  
    d. <b>train_filtered.geojson</b>: unbalanced training data, which will be used in the second round of training.  
    e. <b>train_balanced.geojson</b>: balanced training data. this is what we will use for the first round of training.  

    *If you would like to work through training PoolNet from here and have access to the original pansharpened image (id = 1040010014800C00), sample training and testing geojsons can be found in the 'shapefiles' directory.*

3. Create standardized polygons as uniformly-sized chips for input into the net. Use a batch size that will fit into memory if you will not be training on a generator.  

        data_generator = de.get_iter_data('train_balanced.geojson', batch_size=10000, max_chip_hw=125, normalize=True)  
        x, y = data_generator.next()  

    This will produce chips with only the polygon pixels zero padded to the maximum acceptable chip side dimensions.  

4. Train PoolNet on balanced training data:  

        from pool_net import PoolNet
        p = PoolNet(input_shape = (3,125,125), batch_size = 32)
        p.fit_xy(X_train = x, Y_train = y, save_model = model_name, nb_epoch=15)
        # saves model architecture and weights to model_name.json and model_name.h5 (respectively)

5. Retrain the final dense layer of the network on unbalanced classes:  

        unbal_generator = de.get_iter_data('train_filtered.geojson', batch_size=5000, max_chip_hw=125, normalize=True, nb_epoch = 5)
        x, y = unbal_generator.next()
        # Creates unbalanced training data

        p.retrain_output(X_train=x, Y_train=y)  

6. Load a previously trained model:

        p = PoolNet(input_shape = (3,125,125), batch_size = 32, load_model=True, model_name = model_name)
        p.model.load_weights('model_weighs')


**Note**: The reason why we initially train on balanced data is to allow the model to learn distinct attributes of pools. Given that only about 6% of the original polygons contain pools, training on unbalanced classes would result in the model classifying everything as no pool. Once the model has learned to detect pools in balanced data, we retrain only the final dense layer of PoolNet to minimize the false positives that result from the balanced data training phase.  

## Data Selection and Training  

There were several challenges associated with making an effective net. To minimize the overwhelming and time-consuming task of selecting from the infinite architectures and parameters for PoolNet, we elected to use VGG-16, a model that has been shown to be an effective classifier on previous datasets. We instead focused on parameters such as training/batch size, number of epochs and learning rate as well as data selection and manipulation methods.

### Image Preprocessing

As mentioned [before](#prepare-shapefile-for-training), we elected to zero-pad each chip outside of the polygon. The purpose of this is two-fold: firstly to eliminate any surrounding objects from neighbors that could cause a false positive, and also to standardize the shape and general composition of the input data, as is necessary for convolutional neural networks. Additionally, we ensure that the pixel intensity data is normalized (between 0 and 1) by dividing each pixel by 255.  

<img alt='sample chips after processing' src='images/chips.png' width=700>  
<sub> Sample chips used as input for PoolNet. Notice that only the contents of the polygon are being input to the net.</sub>

### Two-Phase Training

It was important to find a balance between false negatives and false positives when it comes to classifying the chips. As discussed above, the first round of training takes place on balanced classes in order to allow the net to learn based on the composition of the image, as opposed to statistical probability of encountering a pool. After this round of training the model will give over 90% precision and recall when tested on balanced classes. Testing this model on data that is representative of the original data, however, brings the precision down to around 72%, indicating an unacceptably high rate of non-pool chips being classified as having pools.

To minimize this false positive rate without affecting the way the net identifies a pool we retrain only the output layer on unbalanced classes (~6% pools). This simultaneously preserves the way that the net detects pools, while increasing the probability threshold for producing a positive label. There is a fine balance for how much data should be fed into each round of training as well as how many epochs is ideal for each. After playing around with these two variables the net produced the best results when training on more chips in the first round than the second.  

### Misclassified Polygons

After two training phases precision and recall reached approximately 86% and 87%, respectively. Upon manual inspection of the results a few of the causes of misclassification became apparent. Firstly, swimming pools that are partially covered by trees or a tarp, empty, small, or with green water were often falsely classified as 'no pool'. However, the ground truth also appeared to have some incorrectly classified polygons, which the model was actually classifying correctly, despite being marked as a false negative.  

Similarly, a large portion of the geometries that were marked as false positives were actually incorrectly labeled polygons that do have pools. Genuine false positives were usually due to a bright blue object in the back yard with a similar color to many pools. See below for some examples of polygons falsely classified by PoolNet.  

<img alt='Swimming pools not detected by PoolNet.' src='images/missed.png' width=700>  
<sub> Samples of pools that the net misclassified. Notice that many are difficult to see, covered by trees, unusually dark or at the edge of the bounding box. </sub>  

## Results

The current top model was trained first on 9000 polygons with balanced classes (+1000 for validation) for 15 epochs, followed by 20 epochs on 4500 unbalanced classes. Testing this model on the ground truth data gives a precision of 83% and recall of 92%. Given that the ground truth data appears to be flawed, however, we needed a method for getting accurate metrics. To accomplish this we classified 1650 test polygons manually, using multiple sources to confirm the true classification of the polygon. We then compared the results to the original ground truth as well as PoolNet classifications. The new ground truth data gave a precision of 88% and recall of 93%. Results are summarized in the table below.

### Original Data  

<img src='images/Original_results.png' width=350>  
<sub> Results of pool classification based on the original (flawed) 'ground truth' data </sub>

### Ground Truth   

<img src='images/Updated_gt.png' width=350>  
<sub> Results of classification based on the accurate ground truth data </sub>


Check back for future results as we continue to improve the model.

## Docs  
### mltools.pool_net.PoolNet  
<i> class </i> mltools.pool_net. **PoolNet** ( <i> nb_classes=2, batch_size=32, input_shape=(3,125,125), n_dense_nodes=2048, fc=False, vgg=True, load_model=False, model_name=None, train_size=10000 </i> ) [[source](github.com/poolnet_class)]  
__________________________________________________________________________________________
#### About  
A convolutional neural network for pool detection in polygons.  

This classifier can be trained on polygons to detect pools in the property. This model uses [stochastic gradient descent](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/) to optimize [categorical crossentropy](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function).  

| **Parameter:**  | Description:                                                     |
|-----------------|------------------------------------------------------------------|
| nb_classes | int, Number of different image classes. Use 2 for pools. |
| batch_size | int, Number of chips per batch. Defaults to 32. |
| input_shape | tuple(ints), Shape of three-dim input image. Defaults to (3,125,125). Use Theano dimensional ordering. |
| n_dense_nodes | int, Number of nodes to use in dense layers of net. Only applicable if using basic net (not recommended). |
| fc   | bool, Use a fully convolutional model, instead of classic convolutional model. Defaults to False. |
| vgg       | bool, Use VGGNet architecture. Currently gives the best results, defaults to True. |
| load_model | bool, Use a saved trained model (model_name) architecture and weights. Defaults to False. |
| model_name | string, Only relevant if load_model is True. Name of model (not including file extension) to load. Defaults to None (will not load a model) |
| train_size | int, Number of samples to train on per epoch if training on a generator. Defaults to 10000. |  
__________________________________________________________________________________________
#### Methods  

1. [**make_fc_model**](#make_fc_model): Create a fully convolutional version of the current model.  
2. [**fit_xy**](#fit_xy): Train PoolNet on X and Y.  
3. [**fit_generator**](#fit_generator): For training PoolNet on more chips than will fit into memory.  
4. [**retrain_output**](#retrain_output): Retrain only the final layer of the model. For use on unbalanced data.  
5. [**save_model**](#save_model): Save the model architecture as a json and weights from final epoch.  
6. [**load_model_weights**](#load_model_weights): Use a previously trained and saved architecture and weights.  
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
\(train_shapefile, batches=10000, batches_per_epoch=5, min_chip_hw=30, max_chip_hw=125, validation_split=0.1, save_model=None, nb_epoch=15)  
 Train PoolNet on the mltools data generator ([data_extractors.get_iter_data](https://github.com/kostasthebarbarian/mltools/blob/master/mltools/data_extractors.py)).  

|Input| Description |
|---------------|------|
|train_shapefile | string, filepath to shapefile containing polygons to train model on (not including extension)|
|batches | int, number of chips to generate per batch of training. This must fit in memory. |
|batches_per_epoch | int, number of batches to generate and train on per epoch. Total number of chips trained on = *batches x batches_per_epoch* |
|min_chip_hw | int, minimum acceptable side-dimension shape for each polygon. |
|max_chip_hw | int, maximum acceptable side-dimension shape for each polygon. |
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

##### save_model  
\(model_name)  
Saves model architecture as json and weights as h5py doc.  

|Input| Description |
|---------------|------|
|model_name | string, name under which to save model architecture and weights|
|**Output** |  **Description** |
|model_name.json | model architeture |
|model_name.h5 | model weights |  


##### load_model_weights  
\(model_name)  
Load model architecture and weights. Both files must have the same basename (model_name).  

|Input| Description |
|---------------|------|
|model_name | string, filepath and name under which model architecture and weights are saved, not including extension (.json or .h5)|
|**Output** |  **Description** |
|model | model loaded with weights, ready to classify chips. |  


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
\(shapefile, output_name)  
Create a geojson with results of classification saved as properties for each polygon.  

|Input| Description |
|---------------|------|
|shapefile | string, name of shapefile to classify. Will automatically filter out polygons that are too large (side dimensions larger than input_shape[1]) |
|output_name | string, name under which to save the output file |
|**Output** |  **Description** |
|output_name.gejson | file with polygons with classification results as properties |
