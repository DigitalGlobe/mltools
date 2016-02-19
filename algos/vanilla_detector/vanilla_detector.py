"""
Authors: Kostas Stamatiou, Carsten Tusk
Contact: kostas.stamatiou@digitalglobe.com, ctusk@digitalglobe.com
Name: Vanilla detector 
"""

import json
import pickle
import os
import sys

import gdal, ogr, osr
import numpy as np
from sklearn.ensemble import RandomForestClassifier 


def extract_aoi_data(polygon_file):
    """Extracts pixels for each polygon in polygon_file and 
       returns them as masked numpy arrays.
    
       Args:
           polygon_file (str): the name of the input GeoJSON file. 
           The input GeoJSON file must be a collection of Polygons or 
           Multipolygons. Each feature must have an image_name property which
           is a string indicating the name of the georeferenced image
           corresponding to the feature. 
    
       Yields:
           Feature object and corresponding raster object. 
    """
    
    # use ogr module to load features
    shp = ogr.Open(polygon_file)
    lyr = shp.GetLayer()
    feat_list = np.arange(lyr.GetFeatureCount())
    source_SR = lyr.GetSpatialRef()
    
    # get all image catalog ids
    image_names = [lyr.GetFeature(FID).GetFieldAsString('image_name') 
                         for FID in feat_list]

    # get unique images and counts
    unique_image_names, counts = np.unique(image_names, return_counts = True)
    no_unique_images = len(unique_image_names)
    image_list = range(0, no_unique_images)
    # sort feature list by catalog id    
    feat_sorted = np.argsort(image_names)
    # counts_cumsum[i]:counts_cumsum[i+1] are the feature indices in image i
    counts_cumsum = np.hstack([0, np.cumsum(counts)])
    
    for IID in image_list:

        # get image path
        rasterFile = unique_image_names[IID] 
        
        # Get raster geo-reference info
        raster = gdal.Open(rasterFile)
        nbands = raster.RasterCount
        transform = raster.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        targetSR = osr.SpatialReference()
        targetSR.ImportFromWkt(raster.GetProjectionRef())

        feat_this_image = feat_sorted[counts_cumsum[IID]:counts_cumsum[IID+1]]

        for FID in feat_this_image:
        
            # this feature
            feat = lyr.GetFeature(FID)
             
            coordTrans = osr.CoordinateTransformation(source_SR, targetSR)

            # Reproject vector geometry to same projection as raster
            geom = feat.GetGeometryRef()
            geom.Transform(coordTrans)
     
            # Get extent of feat
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
            target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 
                                                           1, gdal.GDT_Byte)
            target_ds.SetGeoTransform( (xmin, pixelWidth, 0,
                                        ymax, 0, pixelHeight,))
               
            # Create target raster projection 
            target_ds.SetProjection(targetSR.ExportToWkt())
        
            # Rasterize zone polygon to raster
            gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])    
        
            # Read raster as arrays    
            dataraster = raster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
        
            datamask = target_ds.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
        
            # replicate mask for each band
            datamask = np.dstack([datamask for i in range(nbands)])
            datamask = datamask.transpose(2,0,1)
            
            # Mask zone of raster
            zoneraster = np.ma.masked_array(dataraster, np.logical_not(datamask))
        
            yield ( feat, zoneraster )


def simple_feature_extractor(data):
    """Computes a feature vector from a data vector consisting of the mean, std 
       and variance.

       Args:
           data (numpy array): data vector

       Yields:    
           The feature vector [ mean(data), std(data), var(data) ]
    """

    yield [np.mean(data), np.std(data), np.var(data)]       


def train_model(polygon_file, classifier_file):
    """Compute feature vector for each polygon in polygon_file, 
       train random forest classifier and 
       output classifier params in pickle file.

       Args:
           polygon_file (str): the name of the input GeoJSON file. 
           classifier_file (str): the name of the classifier parameter file
                                  with .pickle extension.   
    """          
    features = []
    labels = []
    for (feat, data) in extract_aoi_data(polygon_file, imagery_type):        
        label = feat.GetFieldAsString('class_name')
        for featureVector in simple_feature_extractor(data):
            features.append(featureVector)
            labels.append(label)
            print label, featureVector
            
    # train classifier
    X = np.array( features )
    y = np.array( labels )
    # Using a simple random forest with default parameters
    classifier = RandomForestClassifier()
    # train
    classifier.fit( X, y )
    # store model
    with open(classifier_file,"w") as fh:
        pickle.dump( classifier, fh )


def tfRasterToProj(x,y, geoTransform):
    """Converts coordinates from raster pixel to projected coordinate system"""    
    dfGeoX = geoTransform[0] + geoTransform[1] * x + geoTransform[2] * y;
    dfGeoY = geoTransform[3] + geoTransform[4] * x + geoTransform[5] * y;    
    return dfGeoX, dfGeoY


def sliding_window_detector(rasterFile, classifier, winX, winY, stepX, stepY):
    """
    Apply classifier over rasterFile using window (winX, winY) 
    with step (stepX, stepY).

    Args:
        rasterfile (str): 
    """
    raster = gdal.Open(rasterFile)
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
            xc,yc = tfRasterToProj(x0,y0,geoTransform); ring.AddPoint( xc,yc )
            xc,yc = tfRasterToProj(x0+winX,y0,geoTransform); ring.AddPoint( xc,yc )
            xc,yc = tfRasterToProj(x0+winX,y0+winY,geoTransform); ring.AddPoint( xc,yc )
            xc,yc = tfRasterToProj(x0,y0+winY,geoTransform); ring.AddPoint( xc,yc )            
            xc,yc = tfRasterToProj(x0,y0,geoTransform); ring.AddPoint( xc,yc )
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            # Transform to target SRS
            poly.Transform(coordTrans)

            # Read data            
            data = raster.ReadAsArray(x0, y0, winX, winY).astype(np.float)
            # Classify data. Now this depends on if there is one or many feature vectors being computed
            # handle those cases accordingly, maybe a majority decision, maybe count labels, etc
            for featureVector in computeFeatureVectors(data):
                labels = classifier.predict(featureVector)
            
            if labels[0] != 'clutter':            
                features.append({"Class":labels[0], "geom": poly})
            
            x0 += stepX
        y0 += stepY
    
    return features


def detect(polygon_file, imagery_type, modelFile, outputfile, win, step):
    """
    Deploy classifier using sliding window over AOI.
    """

    # get classifier info
    with open(modelFile,"r") as fh:
        classifier = pickle.load(fh)    
    
    # apply sliding window detector on AOI     
    this_tif = extract_aoi_tif(polygon_file, imagery_type)    
    results = sliding_window_detector(this_tif, classifier, win, win, step, step)  
    
    # remove file if it already exists
    if os.path.exists(outputfile):
        os.remove(outputfile)
    
    # write results in geojson format
    writeResults(results, outputfile)


def writeResults(features, outputfile):
    # set up the driver
    driver = ogr.GetDriverByName("GeoJSON")
    
    # create the data source
    data_source = driver.CreateDataSource(outputfile)
    
    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    
    # create the layer
    layer = data_source.CreateLayer("results", srs, ogr.wkbPolygon)
    
    # Add the fields we're interested in
    field_name = ogr.FieldDefn("Class", ogr.OFTString)
    field_name.SetWidth(24)
    layer.CreateField(field_name)
    #layer.CreateField(ogr.FieldDefn("Score", ogr.OFTReal))
    
    # Process the text file and add the attributes and features to the shapefile
    for f in features:
      # create the feature
      feature = ogr.Feature(layer.GetLayerDefn())
      # Set the attributes using the values from the delimited text file
      feature.SetField("Class", f['Class'])
      #feature.SetField("Score", f['Score'])
    
      # Set the feature geometry using the point
      feature.SetGeometry(f['geom'])
      # Create the feature in the layer (shapefile)
      layer.CreateFeature(feature)
      # Destroy the feature to free resources
      feature.Destroy()
    
    # Destroy the data source to free resources
    data_source.Destroy()            


def main(job_json):
    """Runs the vanilla detector pipeline.
       + Computes feature_vector for each geometry in train_file and trains
         standard random forest classifier; 
         train_file must contain at least two classes: "object of interest" 
         and "clutter".
       + Scans AOI in target_file using sliding window, classifies each tile 
         and generates output_file containing all tiles with class 
         other than "clutter", and a classifier.pickle file.
       train_file, target_file and all images should be in the 
       project directory.
    
    Args:
        job_json (str): A filename ending in .json. 
    """

    dictionary = json.load(open(job_json, 'r'))

    # Read parameters      
    train_file = job["train_file"]
    target_file = job["target_file"]
    output_file = job["output_file"]
    window_size = job["machine"]["params"]["window_size"]
    step_size = job["machine"]["params"]["step_size"]

    print "Train..."
    train_model(train_file, imagery_type, 'classifier.pickle')
    
    print "Apply..."
    detect(target_file, imagery_type, 'classifier.pickle', 
                 output_file, window_size, step_size)
    
    print "Done!"   