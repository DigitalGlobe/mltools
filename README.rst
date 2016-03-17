mltools
=======

A collection of Machine Learning (ML) Tools for object detection and classification on DG imagery.

mltools is MIT licenced.

The purpose of this repository is to enable fast prototyping of object detection and classification solutions.

At the moment, there are four modules:

- data_extractors: functions to get pixels from georeferenced imagery
- feature_extractors: functions to derive feature vectors 
- crowdsourcing: interface with Tomnod to obtain training/test/target data and to write back machine output
- json_tools: functions to manipulate json and geojson files

A ML algorithm (MLA) is a class with train and classify/detect functions. At the moment, the repo contains 
the PolygonClassifier MLA which can classify a set of polygons overlayed on a DG image. 

An MLA is typically employed in a script which:
- retrieves training data from Tomnod
- trains the MLA
- tests the MLA and computes accuracy metrics
- deploys the MLA for detection or classification
- writes the MLA results back to the Tomnod database.

Example scripts can be found under /examples.

DevOps
------------

Start with a fresh Ubuntu EC2 instance.

.. highlights::

   sudo apt-get update

   sudo apt-get upgrade

   sudo apt-get install git python-virtualenv libpq-dev python-dev libatlas-base-dev gfortran libfreetype6-dev libpng-dev
   
   ssh-keygen -t rsa
   
   more .ssh/id_rsa.pub # and copy this key to github.com deploy keys for the mltools repo


Install GDAL

.. highlights::
   
   sudo apt-get install gdal-bin
   
   sudo apt-get install libgdal-dev libgdal1h

Clone the repo:

.. highlights::

   git clone git@github.com:kostasthebarbarian/mltools.git
   
   cd mltools
   
   virtualenv venv
   
   . venv/bin/activate
 
Then install the requirements:

.. highlights::

   pip install -r requirements.txt


Comments
--------

mltools is developed as part of an effort to standardize MLA design and implementation. 

Here is a slide with some ideas:

https://docs.google.com/drawings/d/1tKSgFMp0lLd7Abne8CdOhb1PbdJfgCz5x9XkLwDeET0/edit?usp=sharing

The vision is to employ MLA as part of a Crowd+Machine system along the lines of this document:

https://docs.google.com/document/d/1hf82I_jDNGc0NdopXxW9RkbQjLOOGkV4lU5kdM5tqlA/edit?usp=sharing

Imagery in the format required by a MLA (e.g., pansharpened, multi-spectral or orthorectified) can be obtained with the gbdxtools package (https://github.com/kostasthebarbarian/gbdxtools). You need GBDX credentials to use gbdxtools.
