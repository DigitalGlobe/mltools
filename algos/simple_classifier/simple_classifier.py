'''
Simple classifier.

Authors: Carsten Tusk, Kostas Stamatiou
Contact: ctusk@digitalglobe.com, kostas.stamatiou@digitalglobe.com
'''

import sys
import os
import gdal, ogr, osr
import numpy as np
import json
import geojson

from sklearn.ensemble import RandomForestClassifier 

__version__ = 0.2
__date__ = '2016-02-22'
__updated__ = '2016-02-23'


def extract_pixels(polygon_file, raster_file, geom_sr = None):
    """Extracts pixels for each polygon in polygon_file from raster_file.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           geom_sr (osr object): Geometry spatial reference system (srs). 
                                 If None, defaults to the polygon_file srs. 
       
       Yields:
           Feature object, geometry, and corresponding masked numpy array.
    """

    # Open data
    raster = gdal.Open(raster_file)
    nbands = raster.RasterCount
    
    shp = ogr.Open(polygon_file)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())

    # Get raster geo-reference info
    proj = raster.GetProjectionRef()
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Determine coordinate transformation from feature srs to raster srs
    feature_sr = lyr.GetSpatialRef()
    raster_sr = osr.SpatialReference()
    raster_sr.ImportFromWkt(proj)
    coord_trans = osr.CoordinateTransformation(feature_sr, raster_sr)

    # Determine coordinate transformation from raster srs to geometry srs
    # (this is why: the geometry is derived in the raster srs)
    if geom_sr is None:
        coord_trans_2 = osr.CoordinateTransformation(raster_sr, feature_sr)
    else:
        coord_trans_2 = osr.CoordinateTransformation(raster_sr, geom_sr)

    for FID in featList:

        feat = lyr.GetFeature(FID)
    
        # Reproject vector geometry to same projection as raster
        geom = feat.GetGeometryRef()
        geom.Transform(coord_trans)
 
        # Get extent of feat
        geom = feat.GetGeometryRef()
        if (geom.GetGeometryName() == 'MULTIPOLYGON'):
            count = 0
            pointsX = []; pointsY = []
            for polygon in geom:
                geomInner = geom.GetGeometryRef(count)
                ring = geomInner.GetGeometryRef(0)
                numpoints = ring.GetPointCount()
                for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
                count += 1
        elif (geom.GetGeometryName() == 'POLYGON'):
            ring = geom.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            pointsX = []; pointsY = []
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)

        else:
            sys.exit("ERROR: Geometry needs to be Polygon or Multipolygon")

        # get polygon coordinates
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        poly.Transform(coord_trans_2)
        
        xmin = min(pointsX)
        xmax = max(pointsX)
        ymin = min(pointsY)
        ymax = max(pointsY)
    
        # Specify offset and rows and columns to read
        xoff = int((xmin - xOrigin)/pixelWidth)
        yoff = int((yOrigin - ymax)/pixelWidth)
        xcount = int((xmax - xmin)/pixelWidth)+1
        ycount = int((ymax - ymin)/pixelWidth)+1
    
        # Create memory target raster
        target_ds = gdal.GetDriverByName('MEM').Create('', 
                    xcount, ycount, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform((
            xmin, pixelWidth, 0,
            ymax, 0, pixelHeight,
        ))
           
        # Create target raster projection 
        target_ds.SetProjection(raster_sr.ExportToWkt())

        # Rasterize zone polygon to raster
        gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])    
    
        # Read raster as arrays    
        dataraster = raster.ReadAsArray(xoff, yoff, xcount, ycount)
        dataraster = dataraster.astype(np.float)
    
        datamask = target_ds.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
    
        # replicate mask for each band
        datamask = np.dstack([datamask for i in range(nbands)])
        datamask = datamask.transpose(2,0,1)
        
        # Mask zone of raster
        zoneraster = np.ma.masked_array(dataraster, np.logical_not(datamask))
    
        yield (feat, poly, zoneraster)


def simple_feature_extractor(data):
    """Simple feature extractor.

       Args:
           data (numpy array): Pixel data vector.

       Yields:
           A vector with the mean, std and variance of data.
    """

    yield [ np.mean(data), np.std(data), np.var(data) ]


def train_model(polygon_file, raster_file, classifier):
    """Train classifier and output classifier parameters.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           classifier (object): Instance of one of many supervised classifier
                                classes supported by scikit-learn.
       
       Returns:
           Trained classifier (object).                                             
    """
    # compute feature vectors for each polygon
    features = []
    labels = []
    for (feat, poly, data) in extract_pixels(polygon_file, raster_file):        
        label = feat.GetFieldAsString('class_name')
        for featureVector in simple_feature_extractor(data):
            features.append(featureVector)
            labels.append( label )
            print label, featureVector
            
    # train classifier
    X, y = np.array(features), np.array(labels)
    # train
    classifier.fit( X, y )
    # # store model
    # with open(classifier_file,"w") as fh:
    #     pickle.dump( classifier, fh )
    print 'Done!'    
    return classifier


def classify(polygon_file, raster_file, classifier):
    """Deploy classifier and output corresponding list of labels.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           classifier (object): Instance of one of many supervised classifier
                                classes supported by scikit-learn.
       
       Returns:
           List of labels (list).                                             
    """

    # compute feature vectors for each polygon
    labels = []
    for (feat, poly, data) in extract_pixels(polygon_file, raster_file):        
        for featureVector in simple_feature_extractor(data):
            labels_this_feature = classifier.predict(featureVector)                        
        labels.append(labels_this_feature[0])

    print 'Done!'    
    return labels     

        
def write_labels(labels, polygon_file, output_file):
    """Adds labels to target_file to create output_file.
       The number of labels must be equal to the number of features in 
       polygon_file.

       Args:
           labels (list): Label list. 
           polygon_file (str): Filename. Collection of unclassified 
                               geometries in geojson or shp format.
           output_file (str): Output filename (extension .geojson)
    """

    # get input feature collection
    with open(polygon_file) as f:
        feature_collection = geojson.load(f)

    features = feature_collection['features']
    no_features = len(features)
    
    # enter label information
    for i in range(0, no_features):
        feature, label = features[i], labels[i]
        feature['properties']['class_name'] = label

    feature_collection['features'] = features    

    # write to output file
    with open(output_file, 'w') as f:
        geojson.dump(feature_collection, f)     

    print 'Done!'    

  
def main(job_file):
    """Runs the simple_lulc workflow.

       Args:
           job_file (str): Job filename (.json, see README of this repo) 
    """    
   
    # get job parameters
    job = json.load(open(job_file, 'r'))
    image_file = job["image_file"]
    train_file = job["train_file"]
    target_file = job["target_file"]
    output_file = job["output_file"]

    # Using a simple random forest with default parameters 
    # for this demonstration
    classifier = RandomForestClassifier()
        
    print "Train model"
    trained_classifier = train_model(train_file, image_file, classifier)
    
    print "Classify"
    labels = apply_model(target_file, image_file, trained_classifier)
                                        
    print "Write results"    
    write_results(labels, target_file, output_file)

    print "Done!"
   





