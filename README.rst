=======
mltools 
=======

.. image:: https://badge.fury.io/py/mltools.svg
    :target: https://badge.fury.io/py/mltools

A collection of Machine Learning (ML) Tools for object detection and classification on DG imagery.

mltools is MIT licenced.

The purpose of this repository is to enable fast prototyping of object detection and classification solutions.

At the moment, there are four modules:

- data_extractors: functions to get pixels from georeferenced imagery;
- feature_extractors: functions to derive feature vectors; 
- crowdsourcing: interface with Tomnod to obtain training/test/target data and to write machine output to Tomnod DB;
- json_tools: functions to manipulate json and geojson files.

A ML algorithm (MLA) is a class with train and classify/detect functions. At the moment, the repo contains 
the PolygonClassifier MLA which can classify a set of polygons overlayed on a DG image. 

An MLA is typically employed in a script which:
- retrieves training data from Tomnod
- trains the MLA
- tests the MLA and computes accuracy metrics
- deploys the MLA for detection or classification
- writes the MLA results back to the Tomnod database.

Example scripts can be found under /examples.


Installation/Usage
------------------

Start with a fresh Ubuntu EC2 instance:

.. highlights::

   sudo apt-get update

   sudo apt-get upgrade

   sudo apt-get install git python-virtualenv libpq-dev python-dev libatlas-base-dev gfortran libfreetype6-dev libpng-dev
   
Install GDAL drivers:

.. highlights::
   
   sudo apt-get install gdal-bin
   
   sudo apt-get install libgdal-dev

Note: This should install gdal version 1.10.1 for which pygdal will work. Confirm that this is the case with the command:

.. highlights::

   gdal-config --version

If for whatever reason you have another version of gdal you might run into problems.   

Create a python virtual environment in your project directory:

.. highlights::

   cd my_project

   virtualenv venv
   
   . venv/bin/activate

Install GDAL:

.. highlights::

   pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
 
Install mltools:

.. highlights::

   pip install mltools 

You can now use the scripts found in /examples or create your own. Keep in mind 
that the imagery has to be in your project folder. Imagery in the format required by a MLA (e.g., pansharpened, multi-spectral or orthorectified) can be obtained with the gbdxtools package 
(https://github.com/kostasthebarbarian/gbdxtools). You need GBDX credentials to use gbdxtools.
 

DevOps
------

Clone the repo:

.. highlights::

   git clone git@github.com:kostasthebarbarian/mltools.git
   
   cd mltools
   
   virtualenv venv
   
   . venv/bin/activate
 

Generate key:

.. highlights::
   
   ssh-keygen -t rsa
   
   more .ssh/id_rsa.pub 

Copy this key to github.com deploy keys for the mltools repo.


Install the requirements:

.. highlights::

   pip install -r requirements.txt


Comments
--------

mltools is developed as part of an effort to standardize MLA design and implementation. 

Here is a slide with some ideas:

https://docs.google.com/drawings/d/1tKSgFMp0lLd7Abne8CdOhb1PbdJfgCz5x9XkLwDeET0/edit?usp=sharing

The vision is to employ MLA as part of a Crowd+Machine system along the lines of this document:

https://docs.google.com/document/d/1hf82I_jDNGc0NdopXxW9RkbQjLOOGkV4lU5kdM5tqlA/edit?usp=sharing
