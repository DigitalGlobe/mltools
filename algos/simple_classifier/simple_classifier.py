'''
Simple LULC.

This code is meant to illustrate the general workflow of 
a classification engine.

Please keep in mind that writing an actually well performing classification 
engine is beyond the scope of this demo.
Things are kept very very simple for illustrative purposes.

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
__date__ = '2016-02-19'
__updated__ = '2016-02-19'


def extract_pixels(polygon_file, raster_file):
    """Extracts pixels for each polygon in polygon_file from raster_file.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
       
       Yields:
           Feature object and corresponding masked numpy array.
    """

    # Open data
    raster = gdal.Open(raster_file)
    nbands = raster.RasterCount
    
    shp = ogr.Open(polygon_file)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())

    # Get raster geo-reference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)
    
    for FID in featList:

        feat = lyr.GetFeature(FID)
    
        # Reproject vector geometry to same projection as raster
        geom = feat.GetGeometryRef()
        geom.Transform(coordTrans)
 
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
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster.GetProjectionRef())
        target_ds.SetProjection(raster_srs.ExportToWkt())
    
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
    
        yield (feat, zoneraster)


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
           classifier (object): instance of one of many supervised classifier
                                classes supported by scikit-learn
       
       Returns:
           Trained classifier (object).                                             
    """
    # compute feature vectors for each polygon
    features = []
    labels = []
    for (feat,data) in extract_pixels(polygon_file, raster_file):        
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
    return classifier

    print 'Done!'    


def tf_raster_to_proj(x,y, geoTransform):
    """Converts coordinates from raster pixel to projected coordinate system.

       Args:
           x (float): x coordinate.
           y (float): y coordinate.
           geoTransform (float tuple): Geotransform of gdal.Dataset.

       Returns:
           New x coordinate (float) and y coordinate (float).
    """    
    dfGeoX = geoTransform[0] + geoTransform[1] * x + geoTransform[2] * y;
    dfGeoY = geoTransform[3] + geoTransform[4] * x + geoTransform[5] * y;    
    return dfGeoX, dfGeoY


def sliding_window_classifier(raster_file, classifier, 
                              winX, winY, stepX, stepY):
    """Applies classifier on raster_file.

       Args: 
           raster_file (str): Image filename.
           classifier (object): instance of one of many supervised classifier
                                classes supported by scikit-learn
           winX (int): window x size in pixels.
           winY (int): window y size in pixels.
           stepX (int): step x size in pixels.
           stepY (int): step y size in pixels.

       Returns:
           Feature list. Each feature is a dictionary with 
           a class_name key and a geometry key.    
    """
    
    # in case classifier is in pickle format
    # # get classifier parameters
    # with open(modelfile,"r") as fh:
    #     classifier = pickle.load(fh)

    # Open data
    raster = gdal.Open(raster_file)
    nbands = raster.RasterCount
    w = raster.RasterXSize
    h = raster.RasterYSize
    
    # set up coordinate transformations
    geoTransform = raster.GetGeoTransform()
    sourceSRS = osr.SpatialReference()
    sourceSRS.ImportFromWkt( raster.GetProjectionRef() )
    targetSRS = osr.SpatialReference()
    targetSRS.ImportFromEPSG(4326)    
    coordTrans = osr.CoordinateTransformation(sourceSRS,targetSRS)

    features = []

    # simple sliding detection window
    y0 = 0
    while h-y0 >= winY:
        x0 = 0
        while w-x0 >= winX:
            # Create geometry
            ring = ogr.Geometry(ogr.wkbLinearRing)
            xc,yc = tf_raster_to_proj(x0,y0,geoTransform)
            ring.AddPoint( xc,yc )
            xc,yc = tf_raster_to_proj(x0+winX,y0,geoTransform)
            ring.AddPoint( xc,yc )
            xc,yc = tf_raster_to_proj(x0+winX,y0+winY,geoTransform)
            ring.AddPoint( xc,yc )
            xc,yc = tf_raster_to_proj(x0,y0+winY,geoTransform)
            ring.AddPoint( xc,yc )            
            xc,yc = tf_raster_to_proj(x0,y0,geoTransform)
            ring.AddPoint( xc,yc )
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # Transform to target SRS
            poly.Transform(coordTrans)

            # Read data            
            data = raster.ReadAsArray(x0, y0, winX, winY).astype(np.float)
            # Classify data. Now this depends on if there is one or many 
            # feature vectors being computed
            # handle those cases accordingly, maybe a majority decision, 
            # maybe count labels, etc
            for featureVector in simple_feature_extractor(data):
                labels = classifier.predict(featureVector)
                        
            features.append({"class_name":labels[0], "geometry": poly})
            
            x0 += stepX
        y0 += stepY
    
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
    window_size = job["params"]["window_size"]
    step_size = job["params"]["step_size"]
    image_file = job["image_file"]
    train_file = job["train_file"]
    output_file = job["output_file"]

    # Using a simple random forest with default parameters 
    # for this demonstration
    classifier = RandomForestClassifier()
        
    print "Train model"
    trained_classifier = train_model(train_file, image_file, classifier)
    
    print "Apply model"
    results = sliding_window_classifier(image_file, trained_classifier,
                                        window_size, window_size, 
                                        step_size, step_size)

    print "Write results"    
    
    # if file exists, remove 
    if os.path.exists(output_file):
        os.remove(output_file)

    write_results(results, output_file)

    print "Done!"
   





