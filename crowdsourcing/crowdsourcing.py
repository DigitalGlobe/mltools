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

def to_geojson(schema, cat_id, class_name, thres, 
	                              max_number, output_file):
	"""Read features from Tomnod campaign and return geojson.

       Args:
           schema (str): Campaign schema.
           cat_id (str): Image catalog id.
           class_name (str): Feature class (type in Tomnod jargon) name.
           thres(float): Score threshold (only features with score>=thres will
                         be read.
           max_number (int): Maximum number of features to be read.
           output_file (str): Output file name (extension .geojson).              
	"""

	print 'Retrieve data for: ' 
	print 'Schema: ' + schema
	print 'Catalog id: ' + cat_id
	print 'Class name: ' + class_name

	query = """SELECT feature.id, feature.feature
	           FROM {}.feature, tag_type, overlay
	           WHERE feature.type_id = tag_type.id
	           AND feature.overlay_id = overlay.id
	           AND overlay.catalogid = '{}'
	           AND tag_type.name = '{}'
	           AND feature.score >= {}
	           ORDER BY feature.score DESC LIMIT {}""".format(schema, 
	           	                                              cat_id, 
	           	                                              class_name, 
	           	                                              thres, 
	           	                                              max_number)

	
	train_data = DB.db_fetch_array(query)

	# convert to GeoJSON
	geojson_features = [] 
	for entry in train_data:
		feature_id, coords_in_hex = entry
		coords = list(loads(coords_in_hex, hex=True).exterior.coords)
		geojson_feature = geojson.Feature(geometry = geojson.Polygon(coords), 
			                              properties={"id": feature_id, 
			                                          "class_name": class_name, 
			                                          "image_name": cat_id})
		geojson_features.append(geojson_feature)
	feature_collection = geojson.FeatureCollection(geojson_features)	

	# store
	with open(output_file, 'wb') as f:
		geojson.dump(feature_collection, f)		 	   

	print 'Done!'


def create_classify_deploy_geojson(schema, job_id, max_number):
	"""
	Retrieve up to max_number features from schema.feature and store in schema/job_id/deploy_data.json.
	The features are ordered by increasing score, where NULLS go first.
	Only features for which is_priority = True are considered.
	Returns 1 if features were retrieved and saved, else 0.
	For a description of the GeoJSON format, look here: http://geojson.org/geojson-spec.html
	@param schema: the campaign schema in the Tomnod DB
	@param job_id: the machine job id
	@param max_number: maximum number of features
	"""

	print 'Retrieve deploy data for ' + schema

	query = """SELECT id, feature, type_id, imagery_reference
	           FROM {}.feature
	           WHERE is_priority = True	           
	           ORDER BY score NULLS FIRST
	           LIMIT {}""".format(schema, max_number)

	try:
		deploy_data = DB.db_fetch_array(query)
	except:
		print 'Error reading from Tomnod DB. Quitting!'
		return 0

	# convert to GeoJSON
	geojson_features = [] 
	for entry in deploy_data:
		feature_id, coords_in_hex, class_id, cat_id = entry
		coords = list(loads(coords_in_hex, hex=True).exterior.coords)
		geojson_feature = geojson.Feature(geometry=geojson.Polygon(coords), properties={"id": feature_id, "class_id": class_id, "cat_id": cat_id})
		geojson_features.append(geojson_feature)
	feature_collection = geojson.FeatureCollection(geojson_features)	

	# store
	filename = os.path.join(schema, str(job_id), 'deploy_data.json')
	with open(filename, 'wb') as f:
		geojson.dump(feature_collection, f)		 	   

	print 'Done retrieving deploy data for ' + schema
	return 1



def write_geojson_to_gbdx():
	"Writes geojson to gbdx"

	print "Nothing to see here folk"



def read_classified_from_gbdx(schema, job_id, batch_size, machine_id):
	"""
	Dump schema/job_id/deploy_data_classified.json into the Tomnod DB.
	For a description of the GeoJSON format, look here: http://geojson.org/geojson-spec.html	
	@param schema: the campaign schema in the Tomnod DB
	@param job_id: the machine job id 
	@param batch_size: size of result batch written to DB
	@param machine_id: the id of the Machine 'user' in the Tomnod DB
	"""	

	print 'Writing classified deploy data to ' + schema	

	filename = os.path.join(schema, str(job_id), 'deploy_data_classified.json')
	with open(filename, 'r') as f:
		feature_collection = geojson.load(f)

	total_query, counter = "", 1
	features = feature_collection["features"]
	number_of_features = len(features)

	for feature in features:

		feature_id = feature["properties"]["id"]
		class_id = feature["properties"]["class_id"]
		cat_id = feature["properties"]["cat_id"]

		# upsert --- if the Machine has already classified a given feature, then replace previous classification
		update_query = """UPDATE {}.polygon_vote SET tagger_id = {}, type_id = {}
				   		  WHERE polygon_id = {};""".format(schema, machine_id, class_id, feature_id)
		insert_query = """INSERT INTO {}.polygon_vote (polygon_id, tagger_id, type_id) 
				   		  SELECT {}, {}, {}
				   	      WHERE NOT EXISTS 
				   	      (SELECT 1 
				   	       FROM {}.polygon_vote 
				   	       WHERE polygon_id = {} AND tagger_id = {});""".format(schema, feature_id, machine_id, class_id, schema, feature_id, machine_id)
				  
		total_query += update_query + insert_query
		
		if (counter%batch_size == 0) or (counter == number_of_features):
		    DB.db_query(total_query)
		    total_query = ""

		counter += 1	

	print 'Done writing classified deploy data to ' + schema	



	