'''
Simple classifier.

@authors:    Carsten Tusk, Kostas Stamatiou
@copyright:  2016 DigitalGlobe Inc. All rights reserved.
@contact:    ctusk@digitalglobe.com, kostas.stamatiou@digitalglobe.com
'''

import sys
import os
import gdal, ogr, osr
import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier 

__version__ = 0.1
__date__ = '2016-02-22'
__updated__ = '2016-02-22'


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


def apply_model(polygon_file, raster_file, classifier):
    """Deploy classifier and output list of classified features.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           classifier (object): Instance of one of many supervised classifier
                                classes supported by scikit-learn.
       
       Returns:
           List of classified features (list).                                             
    """

    # set target spatial reference to EPSG:4326
    target_sr = osr.SpatialReference().ImportFromEPSG(4326)

    # compute feature vectors for each polygon
    features = []
    for (feat, poly, data) in extract_pixels(polygon_file, raster_file):        
        for featureVector in simple_feature_extractor(data):
            labels = classifier.predict(featureVector)                        
        features.append({"class_name":labels[0], "geometry": poly})

    print 'Done!'    
    return features        

    
def write_results(features, output_file):
    """Writes feature list to geojson file.

       Args:
           features (list): Feature list. Each feature is a dictionary with 
                            a class_name key and a geometry key.
           output_file (str): Output filename (extension .geojson)
    """

    # set up the driver
    driver = ogr.GetDriverByName("GeoJSON")
    
    # create the data source
    data_source = driver.CreateDataSource(output_file)
    
    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    
    # create the layer
    layer = data_source.CreateLayer("results", srs, ogr.wkbPolygon)
    
    # Add the fields we're interested in
    field_name = ogr.FieldDefn("class_name", ogr.OFTString)
    field_name.SetWidth(24)
    layer.CreateField(field_name)
    #layer.CreateField(ogr.FieldDefn("score", ogr.OFTReal))
    
    # Process the text file and add the attributes and features to the shapefile
    for f in features:
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField("class_name", f['class_name'])
        #feature.SetField("score", f['score'])

        # Set the feature geometry using the point
        feature.SetGeometry(f['geometry'])

        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)

        # Destroy the feature to free resources
        feature.Destroy()
    
    # Destroy the data source to free resources
    data_source.Destroy()            
    
  
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
    
    print "Apply model"
    results = apply_model(target_file, image_file, trained_classifier)
                                        
    print "Write results"    
    
    # if file exists, remove 
    if os.path.exists(output_file):
        os.remove(output_file)

    write_results(results, output_file)

    print "Done!"
   





