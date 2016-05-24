# Pool Detection Using Deep Learning

## Table of Contents
1. [PoolNet Workflow](#poolnet-workflow)
2. [Setting up your EC2 Instance](#setting-up-anec2-instance-with-theano)

## PoolNet Workflow

Start with a geojson shapefile and associated tif images.

1. create a geojson with balanced classes for training data (mltools.geojson_tools.create_balanced_geojson)
2. create a shuffled version of (1) for validation data
3. create iterators of polygons of appropriate size zero-padded to input shape


## Setting up an EC2 Instance With Theano  

Set up a gxl2.2 EC2 GPU ubuntu instance on AWS

Follow steps 1 - 9 on [this tutorial](http://markus.com/install-theano-on-aws/).

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
