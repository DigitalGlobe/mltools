"""
Contains functions for reading from and writing to the Tomnod DB.

Author: Kostas Stamatiou
Contact: kostas.stamatiou@digitalglobe.com
"""

import os
import geojson
import json
import numpy as np
from shapely.wkb import loads
import tomnodDB as DB


def to_geojson(schema, 
	           cat_id, 
	           class_name=None, 
	           min_score=0.0, 
	           max_score=1.0, 
	           max_number, 
	           output_file):
	"""Read features from Tomnod campaign and return geojson.

       Args:
           schema (str): Campaign schema.
           cat_id (str): Image catalog id.
           class_name (str): Feature class (type in Tomnod jargon) name.
                             Default is None in which case class is ignored.
           min_score(float): Only features with score>=min_score will be read.
           max_score(float): Only features with score<=max_score will be read.
           max_number (int): Maximum number of features to be read.
           output_file (str): Output file name (extension .geojson).              
	"""

	print 'Retrieve data for: ' 
	print 'Schema: ' + schema
	print 'Catalog id: ' + cat_id
	print 'Class name: ' + class_name

	if class is not None:
		query = """SELECT feature.id, feature.feature, feature.type_id
			       FROM {}.feature, tag_type, overlay
			       WHERE feature.type_id = tag_type.id
			       AND feature.overlay_id = overlay.id
			       AND overlay.catalogid = '{}'
			       AND tag_type.name = '{}'
			       AND feature.score >= {}
			       AND feature.score <= {}
			       ORDER BY feature.score DESC LIMIT {}""".format(schema, 
			           	                                          cat_id, 
			           	                                          class_name, 
			           	                                          min_score,
			           	                                          max_score, 
			           	                                          max_number)
	else:
		 query = """SELECT feature.id, feature.feature, feature.type_id
		            FROM {}.feature, overlay
		            AND feature.overlay_id = overlay.id
		            AND overlay.catalogid = '{}'
		            AND feature.score >= {}
		            AND feature.score <= {}
		            ORDER BY feature.score DESC LIMIT {}""".format(schema, 
		           	                                               cat_id,  
		           	                                               min_score,
		           	                                               max_score, 
		           	                                               max_number)          

	data = DB.db_fetch_array(query)

	# convert to GeoJSON
	geojson_features = [] 
	for entry in data:
		feature_id, coords_in_hex, type_id = entry
		polygon = loads(coords_in_hex, hex=True)
		coords = [list(polygon.exterior.coords)]   # the brackets are dictated
		                                           # by geojson format!!! 
		geojson_feature = geojson.Feature(geometry = geojson.Polygon(coords), 
			                              properties={"id": feature_id, 
			                                          "class_name": type_id, 
			                                          "image_name": cat_id})
		geojson_features.append(geojson_feature)
	
	feature_collection = geojson.FeatureCollection(geojson_features)	

	# store
	with open(output_file, 'wb') as f:
		geojson.dump(feature_collection, f)		 	   

	print 'Done!'