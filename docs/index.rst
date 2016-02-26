.. mltools documentation master file, created by
   sphinx-quickstart on Fri Feb 19 09:36:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mltools
=======

(Disclaimer: work in progress...)

A collection of (un)supervised Machine Learning (ML) Tools for object detection and classification on satellite imagery.

mltools is MIT licenced.

Installation will be easy::

pip install mltools


ML Algorithms (MLAs) are implemented using standard ML libraries such as scikit-learn and tensorflow. 
MLAs also utilize open source libraries which can read from and write to georeferenced satellite images such as gdal.

The purpose of this repository is to enable fast prototyping of object detection and classification solutions employing
one of the existing algorithms or by constructing new ones based on the provided modular tools.

The input of a MLA is one or more of the following:

+ one or more images;
+ a job.json specifying the parameters of the MLA;
+ a train.geojson containing a collection of features, each feature consisting of (at least) a geometry, a class and a unique image identifier;
+ a target.geojson containing a collection of geometries, each feature consisting of (at least) a geometry, a class and a unique image identifier;

The output of a MLA is one or more of the following:

+ one or more processed images
+ an output.geojson containing a collection of features, each feature consisting of (at least) a geometry, a class and a unique image identifier;

The vision is to employ MLA as part of a Crowd+Machine system along the lines of this document:

https://docs.google.com/document/d/1hf82I_jDNGc0NdopXxW9RkbQjLOOGkV4lU5kdM5tqlA/edit?usp=sharing

Imagery in the format required by a MLA (e.g., pansharpened, multi-spectral or orthorectified) can be obtained with the gbdxtools package (https://github.com/kostasthebarbarian/gbdxtools). You need GBDX credentials to use gbdxtools.

.. toctree::
   :maxdepth: 2

   api_docs