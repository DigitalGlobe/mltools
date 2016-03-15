mltools
=======

(Disclaimer: work in progress...)

A collection of Machine Learning (ML) Tools for object detection and classification on DG imagery.

mltools is MIT licenced.

The purpose of this repository is to enable fast prototyping of object detection and classification solutions 
employing one of the existing algorithms or by constructing new ones based on the provided modules.

The input of a ML algorithm (MLA) is one or more of the following:

- one or more images;
- a job.json specifying the parameters of the MLA;
- a train.geojson containing a collection of features, each feature consisting of (at least) a geometry, a class and a unique image identifier;
- a target.geojson containing a collection of geometries, each feature consisting of (at least) a geometry, a class and a unique image identifier.

The output of a MLA is one or more of the following:

- one or more processed images
- an output.geojson containing a collection of features, each feature consisting of (at least) a geometry, a class and a unique image identifier.


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


Docs
----

Explanations of the modules and existing MLAs can be found in docs. 


Comments
--------

mltools is developed as part of an effort to standardize MLA design and implementation. 

Here is a slide with some ideas:

https://docs.google.com/drawings/d/1tKSgFMp0lLd7Abne8CdOhb1PbdJfgCz5x9XkLwDeET0/edit?usp=sharing

The vision is to employ MLA as part of a Crowd+Machine system along the lines of this document:

https://docs.google.com/document/d/1hf82I_jDNGc0NdopXxW9RkbQjLOOGkV4lU5kdM5tqlA/edit?usp=sharing

Imagery in the format required by a MLA (e.g., pansharpened, multi-spectral or orthorectified) can be obtained with the gbdxtools package (https://github.com/kostasthebarbarian/gbdxtools). You need GBDX credentials to use gbdxtools.
