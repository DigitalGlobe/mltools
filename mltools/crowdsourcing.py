"""
What:Contains functions for reading from and writing to the Tomnod database.
Author: Kostas Stamatiou
Created: 02/23/2016
Contact: kostas.stamatiou@digitalglobe.com
"""

import geojson
import os
import psycopg2
import gdal, ogr, osr

from shapely.wkb import loads

class DatabaseError(Exception):
    pass

def get_conn(credentials):
	host = credentials['host']
	db = credentials['db']
	user = credentials['user']
	password = credentials['password']
	if user == '' or password == '':
	    raise DatabaseError('Database username or password not set.')

	conn_string = "host=%s dbname=%s user=%s password=%s" % (host, db, 
		                                                     user, password)
	conn = psycopg2.connect(conn_string)
	conn.autocommit = True
	return conn


def db_fetch(sql, credentials):
	conn = get_conn(credentials)
	cursor = conn.cursor()
	try:
	    cursor.execute(sql)
	    r = cursor.fetchall()
	    conn.close()
	    return r
	except psycopg2.ProgrammingError, e:
	    print "Programming error in query: %s" % e
	    conn.close()
	    return


def db_query(sql, credentials):
    conn = get_conn(credentials)
    cursor = conn.cursor
    try:
        cursor.execute(sql)
    except psycopg2.ProgrammingError, e:
        print "Programming error in query: %s" % e
	conn.close()
	return	     


def train_geojson(schema, 
	              cat_id,
	              max_number, 
	              output_file, 
	              class_name,
	              credentials, 
	              min_score = 0.95, 
	              min_votes = 0,
	              max_area = 1e06
	             ):
	"""Read features from Tomnod campaign and write to geojson.
	   The purpose of this function is to create training data for a machine.
	   Features are read from the DB in decreasing score order.
	   All features are from the same image and of the same class.

       Args:
           schema (str): Campaign schema.
           cat_id (str): Image catalog id.
           max_number (int): Maximum number of features to be read.
           output_file (str): Output file name (extension .geojson).
           class_name (str): Feature class (type in Tomnod jargon) name.
           credentials (dict): Dictionary with host, db, user and password. 
           min_score (float): Only features with score>=min_score will be read.
           min_votes (int): Only features with votes>=min_votes will be read.
           max_area (float): Only import features with (area in m2) <= max_area.
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
		       AND feature.num_votes_total >= {}
		       AND ST_Area(feature.feature) <= {}
		       ORDER BY feature.score DESC LIMIT {}""".format(schema, 
		           	                                          cat_id, 
		           	                                          class_name, 
		           	                                          min_score,
		           	                                          min_votes,
		           	                                          max_area,
		           	                                          max_number)

	data = db_fetch(query, credentials)

	# convert to GeoJSON
	geojson_features = [] 
	for entry in data:
		feature_id, coords_in_hex = entry
		polygon = loads(coords_in_hex, hex=True)
		coords = [list(polygon.exterior.coords)]   # the brackets are dictated
		                                           # by geojson format!!! 
		geojson_feature = geojson.Feature(geometry = geojson.Polygon(coords), 
			                              properties={"id": str(feature_id), 
			                                          "class_name": class_name, 
			                                          "image_name": cat_id})
		geojson_features.append(geojson_feature)
	
	feature_collection = geojson.FeatureCollection(geojson_features)	

	# store
	with open(output_file, 'wb') as f:
		geojson.dump(feature_collection, f)		 	   

	print 'Done!'


def target_geojson(schema, 
	               cat_id,
	               max_number, 
	               output_file, 
	               credentials,
	               max_score = 1.0,
	               max_votes = 0,
	               max_area = 1e06 
	              ):

	"""Read features from Tomnod campaign and write to geojson.
       The purpose of this function is to create target data for a machine.
       Features are read from the DB in increasing score order, nulls first.
       (A feature with null score has not been viewed by a user yet.)
	   
       Args:
           schema (str): Campaign schema.
           cat_id (str): Image catalog id.
           max_number (int): Maximum number of features to be read.
           output_file (str): Output file name (extension .geojson).
           credentials (dict): Dictionary with host, db, user and password. 
           max_score (float): Only features with score<=max_score will be read.
		   max_votes (int): Only features with votes<=max_votes will be read.
		   max_area (float): Only import features with (area in m2) <= max_area.
	"""

	print 'Retrieve data for: ' 
	print 'Schema: ' + schema
	print 'Catalog id: ' + cat_id

	query = """SELECT feature.id, feature.feature, tag_type.name
			   FROM {}.feature, tag_type, overlay
		       WHERE feature.type_id = tag_type.id
	           AND {}.feature, overlay
	           AND feature.overlay_id = overlay.id
	           AND overlay.catalogid = '{}'
	           AND feature.score <= {}
	           AND feature.num_votes_total <= {}
	           AND ST_Area(feature.feature) <= {}
	           ORDER BY feature.score ASC NULLS FIRST
	           LIMIT {}""".format(schema, 
	       	                      cat_id,  
	       	                      min_score,
	       	                      max_score,
	       	                      max_area, 
	       	                      max_number)          

	data = db_fetch(query, credentials)

	# convert to GeoJSON
	geojson_features = [] 
	for entry in data:
		feature_id, coords_in_hex, class_name = entry
		polygon = loads(coords_in_hex, hex=True)
		coords = [list(polygon.exterior.coords)]   # the brackets are dictated
		                                           # by geojson format!!! 
		geojson_feature = geojson.Feature(geometry = geojson.Polygon(coords), 
			                              properties={"id": str(feature_id), 
			                                          "class_name": class_name, 
			                                          "image_name": cat_id})
		geojson_features.append(geojson_feature)
	
	feature_collection = geojson.FeatureCollection(geojson_features)	

	# store
	with open(output_file, 'wb') as f:
		geojson.dump(feature_collection, f)		 	   

	print 'Done!'


def write_geojson(schema, table, input_file, batch_size = 1000):
    """Write contents of geojson to database table.
       At the moment, this only works for the feature table
       of a classification campaign.

       Args:
           schema (str): Campaign schema.
           table (str): The table of schema where to write.
           input_file (str): Input file name (extension .geojson).
           batch_size (int): Write batch_size results at a time.

    """

    print 'Write data to: '
    print 'Schema: ' + schema
    print 'Table:' + table

    # get feature data
    shp = ogr.Open(polygon_file)
    lyr = shp.GetLayer()
    no_features = lyr.GetFeatureCount()

    for i in range(no_features):

        # get feature data
        feat = lyr.GetFeature(i)
        feature_id = int(feat.GetFieldAsString('id')) 
        class_name = feat.GetFieldAsString('class_name')
        if class_name == '': continue      # make sure class name is not empty
        score = float(feat.GetFieldAsString('score'))
        tomnod_priority = float(feat.GetFieldAsString('tomnod_priority')) 

        query = """UPDATE {}.feature 
                   SET type_id = (SELECT id FROM tag_type WHERE name = '{}'),
                       score = {}, 
                       priority = {} 
                   WHERE id = {};""".format(schema,
                   	                        class_name,
                   	                        score, 
                   	                        tomnod_priority,
                   	                        feature_id)

	    total_query += query
	    if  (i%(batch_size-1)  == 0) or (i == no_features-1):
	        db_query(total_query)
	        total_query = ""

	print 'Done!'      