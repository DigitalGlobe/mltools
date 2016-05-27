# Pool Detection Using Deep Learning

## Table of Contents
1. [Setting up a Virtual Environment](#setting-up-an-environment)
2. [PoolNet Workflow](#poolnet-workflow)
3. [Setting up your EC2 Instance](#setting-up-anec2-instance-with-theano)

## Setting up an environment

    conda create -n geo python ipython numpy scipy gdal git libgdal=2

## PoolNet Workflow

Start with a geojson shapefile and associated tif images.

1. Filter shapefile for legitimate polygons (mltools.geojson_tools.filter_polygon_size). Use resolution to determing minimum and maximum acceptable side dimensions for polygons (generally between 40 and 224 pixels for pansharpened images).


2. create train and test geojsons with balanced classes (mltools.geojson_tools.create_balanced_geojson)
3. create iterators of polygons of appropriate size zero-padded to input shape (mltools.data_extractors.get_iter_data)
4. train PoolNet on training data generator


## Setting up an EC2 Instance With Theano  

Set up a gxl2.2 EC2 GPU ubuntu instance on AWS

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
