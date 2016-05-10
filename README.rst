=======
mltools 
=======

.. image:: https://badge.fury.io/py/mltools.svg
    :target: https://badge.fury.io/py/mltools

A collection of Machine Learning (ML) Tools for object detection and classification on DG imagery.

mltools is MIT licenced.

The purpose of this repository is to enable fast prototyping of object detection and classification solutions
using training data from DG Crowdsourcing (aka Tomnod).

There are four modules:

- data_extractors: get pixels and metadata from georeferenced imagery; uses geoio (https://github.com/digitalglobe/geoio);
- features: functions to derive features from pixels; 
- crowdsourcing: interface with Tomnod to obtain training/test/target data and to write machine output to Tomnod DB;
- geojson_tools: functions to manipulate geojson files.

A ML algorithm (MLA) is a class with train, test and classify/detect functions. 
Presently, the repo contains the PolygonClassifier MLA which can classify a set of polygon 
geometries in a geojson. The mltools acceptable geojson format is found in /examples. 

An MLA is typically employed in a script which:

1. retrieves training, test and target data from the Tomnod database;
2. trains the MLA;
3. tests the MLA on the test data and computes accuracy metrics;
4. deploys the MLA on the target data for detection or classification;
5. writes the MLA results back to the Tomnod database.

Step 1 can be omitted if data is available from a source other than Tomnod. 
(However, the data must respect the geojson format found in /examples.)
Step 5 can also be omitted if we don't want to write the results back to Tomnod.

Example scripts can be found under /examples. 


Installation/Usage
------------------

For Ubuntu, install conda with the following commands (choose default options at prompt)::

   wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
   bash Miniconda2-latest-Linux-x86_64.sh

   
For OS X, install conda with the following commands (choose default options at prompt)::

   wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
   bash Miniconda2-latest-MacOSX-x86_64.sh

Then run::

   bash

so that modifications in your .bashrc take effect. 

Create a conda environment::

   conda create -n env python ipython numpy scipy gdal git  
   
Activate the environment::

   source activate env

Install mltools::

   pip install mltools

If you had a previous version of mltools installed, then (try)::

   pip install mltools --upgrade --ignore-install   

You can now copy the scripts found in /examples in your project directory or create your own. 
Keep in mind that the imagery has to be in your project folder and it should have the same name as the image_name 
property in the geojson. Imagery in the format required by a MLA (e.g., pansharpened, multi-spectral or orthorectified) can be obtained with the gbdxtools package (https://github.com/digitalglobe/gbdxtools).

To exit your conda virtual environment::

   source deactivate 
 

Development
-----------

Activate the conda environment::

   source activate env

Clone the repo::

   git clone https://github.com/kostasthebarbarian/mltools
   
   cd mltools
   
Install the requirements::

   pip install -r requirements.txt

Please follow this python style guide: https://google.github.io/styleguide/pyguide.html.
80-90 columns is fine.

To exit your conda virtual environment::

   source deactivate


Comments
--------

mltools is developed as part of an effort to standardize MLA design and implementation. 

Here is a slide with some ideas:

https://docs.google.com/drawings/d/1tKSgFMp0lLd7Abne8CdOhb1PbdJfgCz5x9XkLwDeET0/edit?usp=sharing

The vision is to employ MLA as part of a Crowd+Machine system along the lines of this document:

https://docs.google.com/document/d/1hf82I_jDNGc0NdopXxW9RkbQjLOOGkV4lU5kdM5tqlA/edit?usp=sharing
