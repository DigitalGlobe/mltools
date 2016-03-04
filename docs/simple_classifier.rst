Simple Classifier
=================

This is a basic object classifier.
An object consists of a geometry and an underlying georeferenced image.
Simple Classifier classifies a set of objects by extracting 
a simple feature vector for each object and employing a Random Forest Classifier.
Training data consists of a set of labeled objects selected from the same image.  

Input
-----

- job.json

.. code-block:: javascript

   {"params":{"no_trees":100},
    "image_file": "aoi.tif",
    "train_file": "train.geojson",
    "target_file":, "target.geojson",
    "output_file": "output.geojson"}

train_file is the name of the geojson file containing labeled objects on image_file.
target_file is the name of the geojson file containing the objects on image_file to be classified.

- train.geojson

.. code-block:: javascript

   {
    "type": "FeatureCollection",
    "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
    "features": [
      { "type": "Feature", "properties": { "class_name": "boat", "image_name": "aoi.tif" }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 113.994918315430752, 22.129573210547246 ], [ 113.996701993119601, 22.129611161136371 ], [ 113.996512240173971, 22.127789532858394 ], [ 113.99488036484162, 22.127865434036643 ], [ 113.994918315430752, 22.129573210547246 ] ] ] } },
      { "type": "Feature", "properties": { "class_name": "clutter", "image_name": "aoi.tif" }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 113.983495188104271, 22.130635827042735 ], [ 113.984406002243261, 22.13101533293398 ], [ 113.985772223451747, 22.129383457601627 ], [ 113.984064446941133, 22.129497309368997 ], [ 113.984064446941133, 22.129497309368997 ], [ 113.983495188104271, 22.130635827042735 ] ] ] } }]
   }

train.geojson must contain at least two classes. 

- target.geojson

.. code-block:: javascript

   {
    "type": "FeatureCollection",
    "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
    "features": [
      { "type": "Feature", "properties": { "image_name": "aoi.tif" }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 113.994918315430752, 22.129573210547246 ], [ 113.996701993119601, 22.129611161136371 ], [ 113.996512240173971, 22.127789532858394 ], [ 113.99488036484162, 22.127865434036643 ], [ 113.994918315430752, 22.129573210547246 ] ] ] } },
      { "type": "Feature", "properties": { "image_name": "aoi.tif" }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 113.983495188104271, 22.130635827042735 ], [ 113.984406002243261, 22.13101533293398 ], [ 113.985772223451747, 22.129383457601627 ], [ 113.984064446941133, 22.129497309368997 ], [ 113.984064446941133, 22.129497309368997 ], [ 113.983495188104271, 22.130635827042735 ] ] ] } }]
   }


Output
-----

- output.geojson: Contains all objects in target.geojson, classified. 


Requirements
------------

python, numpy, scipy, scikit-learn, gdal


Comments
--------

This algorithm is meant to illustrate the general workflow of a classification engine.
Please keep in mind that writing an actually well performing classification engine is beyond the scope of this package.
Things are kept very very simple for illustrative purposes.

