Simple Land Use Land Cover Classifier
=====================================

This is a simple, supervised Land Use Land Cover (LULC) classifier.
It segments an entire image in tiles and classifies each tile based on a given set of classes.
Training data consists of a set of labeled geometries which have been selected from the same image.  

Input
-----

- job.json

.. code-block:: javascript

   {"params":{"window_size":100, "step_size":100},
   "image_file": "aoi.tif",
   "train_file": "train.geojson",
   "output_file": "output.geojson"}

window_size is the tile size in pixels. step_size is the stride of the window in pixels. train_file is the name 
of the geojson file containing labeled geometries belonging to the image.

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

- output.geojson: contains a collection of all classified tiles. 


Requirements
------------

python, numpy, scipy, scikit-learn, gdal


Comments
--------

This algorithm is meant to illustrate the general workflow of a classification engine.
Please keep in mind that writing an actually well performing classification engine is beyond the scope of this package.
Things are kept very very simple for illustrative purposes.

