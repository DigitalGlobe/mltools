# Pool Detection Using Deep Learning

## Table of Contents

1. [About PoolNet](#about-poolnet)
2. [Getting Started](#getting-started)
    * [Setting up your EC2 Instance](#setting-up-anec2-instance-with-theano)
    * [Setting up a Virtual Environment](#setting-up-an-environment)
3. [PoolNet Workflow](#poolnet-workflow)
4. [Docs](#docs)

## About PoolNet

PoolNet utilizes the [VGG-16](https://arxiv.org/pdf/1409.1556.pdf) network architecture, a 16-layer convolutional neural network, the top-scoring submission for the 2014 [ImageNet](http://www.image-net.org/challenges/LSVRC/2014/) classification challenge.

This network is trained on satellite images of various property polygons in order to classify these properties as ones with or without pools (see image below). This model provides an efficient and reliable way to determine which homes have pools, information that is useful to insurance companies. With appropriate training data this model can be extended to applications such as vehicles, solar panels and buildings.

<img alt='Example property polygons. Red indicates no pool, green indicates that there is a pool within the polygon.' src='images/sample_polygons.png' width=300>  
<sub> Example property polygons. Red indicates no pool, green indicates that there is a pool within the polygon. </sub>

### The challenge:

Pools turn out to be very diverse in satellite images, varying in shape, color, tree-coverage and location. A convolutional neural net is therefore a promising option for learning to detect pools, providing the flexibility of learning common abstract qualities of the item of interest independent of location in the input image. The large amount of parameters trained in PoolNet allows it to learn a variety of features that pools have that other machine learning techniques may overlook.


## Getting Started  

PoolNet requires a GPU to prevent training from being prohibitively slow. Before getting started you will need to set up an EC2 instance with Theano.

### Setting up an environment

1. Create the environment:  

        conda create -n geo python ipython numpy scipy gdal git libgdal=2  

2. upgrade pip:  

        pip install --upgrade pip  

3. install mltools:  

        pip install mltools

### Setting up an EC2 Instance With Theano  

Set up an Ubuntu g2.2xlarge EC2 GPU ubuntu instance on AWS

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


## PoolNet Workflow


Start with a geojson shapefile ('shapefile.geojson') and associated tif images:  

<img alt='Raw shapefile polygons overlayed on tif' src='images/raw_polygons.png' width=500>   
<sub> Pansharpened tif image with associated polygons overlayed. Green polygons indicate there is a pool in the property. </sub>

### Prepare Shapefile for Training

1. Filter shapefile for legitimate polygons. Use resolution to determine minimum and maximum acceptable chip side dimensions (generally between 30 and 125 pixels for pansharpened images).  

    <img alt='Small polygons to be filtered out of shapefile' src='images/small_polygons.png' width=200>
    <img alt='Shapefile with small polygons filtered out' src='images/filtered_polygons.png' width=200>

        import mltools.geojson_tools as gt
        import mltools.data_extractors as de

        gt.filter_polygon_size('shapefile.geojson', 'filtered_shapefile', min_polygon_hw=30, max_polygon_hw=125)
        # creates filtered_shapefile.geojson

2. Create train and test Shapefiles with and without balanced classes.

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

3. Create standardized polygons as uniformly-sized chips for input into the net. Use a batch size that will fit into memory if you will not be training on a generator.  

        data_generator = de.get_iter_data('train_balanced.geojson' batch_size=10000, max_chip_hw=125, normalize=True)
        x, y = data_generator.next()  

    This will produce chips with only the polygon pixels zero padded to the maximum acceptable chip side dimensions.

    <sample images of chips>

4. Train PoolNet on training data:

        from pool_net import PoolNet
        p = PoolNet(input_shape = (3,125,125), batch_size = 32, nb_epoch = 20)
        p.fit_xy(X_train = x, Y_train = y, save_model = model_name)
        # saves model architecture and weights to model_name.json and model_name.h5 (respectively)

## Docs  
### mltools.pool_net.PoolNet  
<i> class </i> mltools.pool_net. **PoolNet** ( <i> nb_epoch=4, nb_classes=2, batch_size=32, input_shape=(3,125,125), n_dense_nodes=2048, fc=False, vgg=True, load_model=False, model_name=None, train_size=10000 </i> ) [[source](github.com/poolnet_class)]  
__________________________________________________________________________________________
#### About  
A convolutional neural network for pool detection in polygons.  

This classifier can be trained on polygons to detect pools in the property. This model uses [stochastic gradient descent](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/) to optimize [categorical crossentropy](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function).  

| **Parameter:**  | Description:                                                     |
|-----------------|------------------------------------------------------------------|
| nb_epoch   | int, Number of epochs to train for. Defaults to 4. |
| nb_classes | int, Number of different image classes. Use 2 for pools. |
| batch_size | int, Number of chips per batch. Defaults to 32. |
| input_shape | tuple(ints), Shape of three-dim input image. Defaults to (3,125,125). Use Theano dimensional ordering. |
| n_dense_nodes | int, Number of nodes to use in dense layers of net. Only applicable if using basic net (not recommended). |
| fc   | bool, Use a fully convolutional model, instead of classic convolutional model. Defaults to False. |
| vgg       | bool, Use VGGNet architecture. Currently gives the best results, defaults to True. |
| load_model | bool, Use a saved trained model (model_name) architecture and weights. Defaults to False. |
| model_name | string, Only relevant if load_model is True. Name of model (not including file extension) to load. Defaults to None (will not load a model) |
| train_size | int, Number of samples to train on per epoch if training on a generator. Defaults to 10000. |  

#### Methods  

[make_fc_model()](#make_fc_model)

##### make_fc_model()  
Re-compile current model as [fully convolutional]. Beheads the standard convolutional model and adds three 2D convolutional layers. 
