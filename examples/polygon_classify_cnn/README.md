# Pool Detection Using Deep Learning

## Table of Contents

1. [About PoolNet](#about-poolnet)
    * [The Challenge](#the-challenge)
    * [Network Architecture](#network-architecture)
2. [Getting Started](#getting-started)
    * [Setting up your EC2 Instance](#setting-up-your-ec2-instance)
    * [Setting up an Environment](#setting-up-an-environment)
3. [PoolNet Workflow](#poolnet-workflow)
    * [Getting the Imagery](#getting-the-imagery)
    * [Prepare Geojsons](#prepare-geojsons)
    * [Training the Network](#training-the-network)
        - [Create the Training Chips](#create-the-training-chips)
        - [First Training Phase](#first-training-phase)
        - [Second Training Phase](#second-training-phase)
    * [Testing the Network](#testing-the-network)
4. [Performance](#performance)
    * [Results](#results)
    * [Misclassified Polygons](#misclassified-polygons)

## About PoolNet

PoolNet uses deep learning with a [Convolutional Neural Network](http://neuralnetworksanddeeplearning.com/chap6.html#introducing_convolutional_networks) (CNN) to classify which satellite images of various property polygons contain a pool. This model provides an efficient and reliable way to identify homes with swimming pools, information that is valuable to insurance companies and would otherwise be challenging to collect at a large scale. With appropriate training data this model can be extended to various applications such as vehicles, boats, solar panels and buildings.

<img alt='Example property polygons. Red indicates no pool, green indicates that there is a pool within the polygon.' src='images/sample_polygons.png' width=400>  
<sub> Example property polygons. Red indicates no pool, green indicates that there is a pool within the polygon. </sub>  

### The Challenge

Pools turn out to be very diverse in satellite images, varying in shape, color, tree-coverage and location. A convolutional neural network is therefore well-suited for machine detection of pools, providing the flexibility to learn common abstract qualities of the item of interest independently of location in the input image. The vast number of parameters trained in PoolNet allows it to learn a variety of features that pools have that other machine learning techniques and even the human eye may overlook.

<img alt='Major challenge in machine classification of pools- diversity of pools in satellite imagery' src='images/pool_diversity.png'>  
<sub>Various pool-containing polygons from the test data. Notice the diversity in shape, size, color, intensity and location in the polygon. This makes machine classification of pools very challenging. </sub>  


### Network Architecture

PoolNet utilizes the [VGG-16](https://arxiv.org/pdf/1409.1556.pdf) network architecture, a 16-layer convolutional neural network and the top-scoring submission for the 2014 [ImageNet](http://www.image-net.org/challenges/LSVRC/2014/) classification challenge. This architecture is composed of many small (3x3) convolutional filters, which enables such a deep network to be trained and deployed in a timely manner on a GPU.  

<b>VGG-16 Architecture</b>  
<img src='images/VGGNet.png' width=500>  
<sub>Architecture of VGGNet. The input layer (blue) is comprised of a 3-channel pansharpened rgb property polygon. Each green layer represents a convolution of the original image with a 3x3 kernel. Max-pooling layers (denoted by the black MP) are used for downsampling the convolved image by taking only the most intense of a given pool of pixels. The yellow layers at the end represent fully connected layers where all remaining pixels are flattened into a 2-dimensional layer of nodes. Finally two nodes of the the softmax layer at the end each represent one class (pool or no pool) and produces a probability the the input image belongs to that class. </sub>  

## Getting Started  

PoolNet should run on a GPU to prevent training from being prohibitively slow. Before getting started you will need to set up an EC2 instance with Theano.

### Setting up your EC2 Instance

Begin by setting up an Ubuntu g2.2xlarge EC2 GPU ubuntu instance on AWS.  

Follow steps 1 - 9 on [this tutorial](http://markus.com/install-theano-on-aws/).  
In short:

1. Update packages:  

    ```bash
    sudo apt-get update  
    sudo apt-get -y dist-upgrade
    ```

2. Open tmux:  

    ```bash
    tmux  
    ```

3. Install dependencies:  

    ```bash
    sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy
    ```

4. Get cuda toolkit (7.0):  

    ```bash
    sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb  
    ```

5. Depackage cuda:  

    ```bash
    sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb  
    ```

6. Add package and install cuda driver (~5 min)  

    ```bash
    sudo apt-get update  
    sudo apt-get install -y cuda  
    ```

7. Add cuda nvcc and ld_library_path to path:  

    ```bash
    echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc  
    ```

8. Reboot:  

    ```bash
    sudo reboot  
    ```

9. Create a file entitled .theanorc in the /home/ubuntu/ directory as follows:  

    <b>.theanorc config: </b>  

    ```bash
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
    ```

### Setting up an Environment

Before training your net be sure to install mltools and activate your conda environment. <b>Follow these [instructions](https://github.com/digitalglobe/mltools#installationusage) before continuing.</b>  

Activate your environment:  

```bash
source activate env
```

**Note**: to run PoolNet you must install the current version of the master branch of mltools, in addition to a couple of other dependencies on your conda environment:  

```bash
pip install git+https://github.com/DigitalGlobe/mltools  
conda install scikit-image  
conda install h5py  
conda install matplotlib
```

## PoolNet Workflow

The steps described here require a geojson file (properties.geojson) with training data, associated pansharpened GeoTiff images (named after their GBDX catalog id), and [pool_net.py](https://github.com/DigitalGlobe/mltools/blob/master/examples/polygon_classify_cnn/pool_net.py) placed in your working directory.

<img alt='Raw geojson polygons overlayed on tif' src='images/raw_polygons.png' width=750>   
<sub> Pansharpened tif image with associated polygons overlayed. Green polygons indicate there is a pool in the property. </sub>

### Getting the Imagery

Order, create and download the pansharpened image with catalog id 1040010014800C00 using [gbdxtools](http://github.com/DigitalGlobe/gbdxtools). Instructions can be found [here](http://gbdxtools.readthedocs.io/en/latest/) and [here](https://github.com/DigitalGlobe/mltools/tree/master/examples/polygon_classify_random_forest). This is the image where we will be classifying property parcels in those that contain swimming pools and those that don't.

### Prepare Geojsons

<b>In this section we will create the following geojsons:</b>

<img alt='Schema for geojsons created from the original raw data.' src='images/repr_geojsons.png' width=200>  

a. <b>properties.geojson</b>: Original file with all polygons.  
b. <b>filtered_geojson.geojson</b>: File with all polygons with side dimensions between 30 and 125 pixels.  
c. <b>test_filtered.geojson</b>: Test data with filtered polygons and unbalanced classes. Don't touch it until testing the model!  
d. <b>train_filtered.geojson</b>: Unbalanced training data, which will be used in the second round of training.  
e. <b>train_balanced.geojson</b>: Balanced training data. This is what we will use for the first round of training.   

We initially filter properties.geojson to get rid of polygons that we have deemed too small or too large. We then create train and test data, as well as a batch of training data with balanced classes, the motivation for which is detailed [below](#first-training-phase). <!--If you do not have access to an original geojson (properties.geojson), there are sample filtered train and test geojsons (test_filtered.geojson, train_filtered.geojson, and train_balanced.geojson), which are sufficient for training and testing the model, so you can omit this section and continue to [Training the Network](#training-the-network). **where are these?**--> Your working directory should contain properties.geojson as well as all image strips referenced in the feature properties of properties.geojson.

<img alt='Small polygons to be filtered out of properties.geojson' src='images/small_polygons.png' height=175>
<img alt='Geojson with small polygons filtered out' src='images/filtered_polygons.png' height=175>

1. **Create filtered_geojson.geojson:**

    <img src='images/repr_geojsons_2.png' width=175>

    Navigate to the directory that contains properties.geojson and the associated tif imagery. Open an iPython terminal and filter your original geojson. Use resolution to determine minimum and maximum acceptable chip size dimensions (generally between 30 and 125 pixels for pansharpened images).  

    ```python
    from mltools import geojson_tools as gt

    gt.filter_polygon_size('properties.geojson', output_file = 'filtered_geojson.geojson',
                           min_side_dim = 30, max_side_dim = 125)
    ```

2. **Create train_filtered.geojson and test_filtered.geojson:**

    <img src='images/repr_geojsons_3.png' width=175>

    ```python
    gt.create_train_test('filtered_geojson.geojson', output_file = 'filtered.geojson',
                         test_size = 0.2)
    ```

3. **Create balanced training data 'train_balanced.geojson':**  

    <img src='images/repr_geojsons_4.png' width=175>

    ```python
    gt.create_balanced_geojson('train_filtered.geojson',
                               classes = ['No swimming pool', 'Swimming pool'],
                               output_file = 'train_balanced.geojson')
    ```


### Training the Network  

We train this network in two rounds to address the issue of class imbalance. The first round takes place on balanced data and the second on the original distribution of classes.


#### First Training Phase

One challenge presented by the data is that only about 6% of the polygons actually contain pools. This class imbalance causes the net to learn only the statistical probability of encountering a pool, and thus produce only 'non-pool' classifications. To force the net to instead learn the general attributes of pools based on image composition, we train it on balanced data (equal number of 'pool' and 'no pool' polygons), which we created in train_balanced.geojson [above](#prepare-geojsons).

During training the model will extract chips from the GeoTiff imagery using polygon features from the geojson file. You can see examples of these chips below:

<img alt='sample chips after processing' src='images/chips.png' width=700>  
<sub> Sample chips used as input for PoolNet. Notice that only the contents of the polygon are being input to the net.</sub>

To train the net we simply create an instance of PoolNet and then pass it the appropriate geojson file from the workflow [above](#prepare-geojsons). In the first round the network is trained on the train_balanced.geojson file.


1. Create a PoolNet instance:

    ```python
    from pool_net import PoolNet

    p = PoolNet(classes = ['No swimming pool', 'Swimming pool'], batch_size = 32,
                input_shape = (3,125,125))
    ```

    This step creates a PoolNet instance with appropriate parameters. The input_shape parameter should be entered as (*n_channels, max chip height, max chip width*).

2. Train the network using the ```fit_from_geojson()``` method:

    ```python
    p.fit_from_geojson(train_geojson = 'train_balanced.geojson', max_side_dim = 125,
                       nb_epoch = 15, min_side_dim = 30, train_size = 10000,
                       save_model = 'my_model')  
    ```

    The final command executes training on the balanced data you created in the previous section. The nb_epoch argument defines how many rounds of training to perform on the network. In general this should be until validation loss stops decreasing to avoid overfitting. Weights for the model will be saved after each epoch in the models directory, so it is possible to roll back the training to earlier epochs if necessary.


#### Second Training Phase

After this round of training the model produces over 90% precision and recall when tested on *balanced* classes. Testing this model on data that is representative of the original data brings the precision down to around 72%, indicating an unacceptably high rate of non-pool chips being classified as having pools. To see these results for yourself you may skip the second round of training and test the current model [below](#testing-the-network).

To minimize the false positive rate without severely reducing recall, we retrain only the output layer of the model on imbalanced classes. This simultaneously preserves the way that the net detects pools, while decreasing the probability the then network will generate a positive label.   

Retrain final layer of network on unbalanced data as follows:  

```python
p.fit_from_geojson(train_geojson = 'train_filtered.geojson', retrain = True,
                   max_side_dim = 125, min_side_dim = 30, train_size = 5000,
                   nb_epoch = 5, save_model = 'my_model_rnd2')  
```


### Testing the Network  

We now have a fully trained network that is ready to be tested. Here we will produce a confusion matrix from 2500 test polygons.  

1. Create test data from test_filtered.geojson:  

    ```python
    from mltools import data_extractors as de

    x, y = de.get_uniform_chips('test_filtered.geojson', num_chips = 2500,
                                max_side_dim = 125, min_side_dim = 30,
                                classes = ['No swimming pool', 'Swimming pool'])
    ```

2. Use model to predict classes of test chips:

    ```python
    y_pred = p.model.predict_classes(x)  
    ```

3. Convert y from one-hot encoding to list of classes:

    ```python
    y_true = [i[1] for i in y]  
    ```

4. Create confusion matrix from y_true and y_pred:  

    ```python
    from sklearn.metrics import confusion_matrix

    print confusion_matrix(y_true, y_pred)

    # Output should appear as follows:
    # [[2324 (true_positives)  15 (false_positives)]
    # [18 (false_negatives)  143 (true_negatives)]]
    ```  

5. Calculate precision and recall:

    ```python
    precision = 143.0 / (143.0 + 15.0)
    recall = 143.0 / (143.0 + 18.0)
    ```


### Visualizing Results  

To visualize the results we must create a new geojson for which each polygon has a PoolNet classification, certainty of that classification and the ground truth classification listed in the properties. Here we will classify all polygons in test_filtered.geojson and save them to test_classed.geojson. We will then use the classified geojson to visualize polygon classifications overlayed on the original tif image (1040010014800C00.tif).  

Complete the first step only if you would like to classify your own data. Otherwise just use test_classed.geojson and continue on to step 2.  

1. Classify all test data (5-10 minutes):

    ```python
    p.classify_geojson('test_filtered.geojson', output_name = 'test_classed.geojson',
                       max_side_dim = 125, min_side_dim = 30, numerical_classes = False)  
    ```

2. Open 1040010014800C00.tif in QGIS:  

    **Layer > Add Layer > Add Raster Layer...**  

    <img src='images/QGIS_step1.png' width=200>  

    **Select appropriate tif image**  

    <img src='images/QGIS_step2.png' width=200> ->
    <img src='images/QGIS_step3.png' width=155>

3. Open test_classed.geojson as a vector file:  

    **Layer > Add Layer > Add Vector Layer...**  
    <img src='images/QGIS_vector.png' width=200>  

    **Select test_classed.png**  
    <img src='images/QGIS_geojson.png' width=200> ->
    <img src='images/QGIS_overlay.png' width=160s>  

4. Color Polygons by Category:

    **Layer > Properties**  
    *Vector layer must be highlighted*  
    <img src='images/QGIS_categorize.png' width=200>  

    **Select Ground Truth Property, click 'Classify'**  
    <img src='images/QGIS_classify.png' width=200> ->
    <img src='images/QGIS_clssed.png' width=200>  

    **Format polygons by class**  
    <img src='images/QGIS_format_polygons.png' width=200> ->
    <img src='images/QGIS_simplefill.png' width=200>  

    **change fill to transparent**  
    <img src='images/QGIS_transparent.png' width=100>
    <idmg src='images/QGIS_border.png' width=100>  

    **Result**  
    <img src='images/QGIS_final.png' width=400>  
    <sub>Classified geojson with polygons colored by ground-truth class.</sub>


## Performance  

Below is an overview of the model's performance on test data.

### Results

The current top model was trained first on 9000 polygons with balanced classes (+1000 for validation) for 15 epochs, followed by 20 epochs on 4500 unbalanced classes. Testing this model on initial test data gives a precision and recall of 83% and 92%, respectively. The original test data, however, appears to be flawed upon visual inspection of results (see [below](#misclassified-polygons)). We therefore needed a method for getting accurate metrics. To accomplish this we classified 1650 test polygons manually, using multiple sources to confirm the true classification of each polygon. We then compared the results to the original test data as well as PoolNet classifications. The reliable test data indicates a precision of 88% and recall of 93% by our model. Results are summarized in the table below.  


#### Test Dataset #1:  

<img src='images/Original_results.png' width=350>  
<sub> Results of pool classification based on the original (flawed) 'ground truth' data </sub>

#### Test Dataset #2:    

<img src='images/Updated_gt.png' width=350>  
<sub> Results of classification based on the accurate ground truth data </sub>


Check back for future results as we continue to improve the model.  

### Misclassified Polygons

Upon manual inspection of the results a few of the causes of misclassification became apparent. Firstly, swimming pools that are partially covered by trees or a tarp, empty, small, or with green water were often falsely classified as 'no pool'. However, the ground truth also appeared to have some incorrectly classified polygons, which the model was actually classifying correctly, despite being marked as a false negative.  

Similarly, a large portion of the geometries that were marked as false positives were actually incorrectly labeled polygons that do have pools. Genuine false positives were usually due to a bright blue object in the back yard with a similar color to many pools. See below for some examples of polygons falsely classified by PoolNet.  

<img alt='Swimming pools not detected by PoolNet.' src='images/missed.png' width=700>  
<sub> Samples of pools that the net misclassified. Notice that many are difficult to see, covered by trees, unusually dark or at the edge of the bounding box. </sub>
