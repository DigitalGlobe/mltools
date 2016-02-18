"""
Vanilla classifier
This is based on Carsten's demo_machine
Features: [np.mean(data), np.std(data), np.var(data)]
Classification algorithm: Random forests
"""

import sys
import os
import pickle
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from reportlab.lib.testutils import outputfile

import gdal, ogr, osr
import numpy as np
from sklearn.ensemble import RandomForestClassifier 


def extract_aoi_data(polygonFile, imagery_type):
    """
    Extracts pixels for each polygon in polygonFile and returns them as masked numpy arrays.
    polygonFile is a GeoJSON which follows the specification in the README of the repo.
    imagery_type is the type of imagery used by the classifier
    """
    
    # get directory where imagery is stored
    images_dir = os.getenv("IMAGES_DIR")

    # use ogr module to load features
    shp = ogr.Open(polygonFile)
    lyr = shp.GetLayer()
    featList = np.arange(lyr.GetFeatureCount())
    sourceSR = lyr.GetSpatialRef()
    
    # get all image catalog ids
    image_catalog_ids = [lyr.GetFeature(FID).GetFieldAsString('cat_id') for FID in featList]
    # get unique image catalog ids and counts
    unique_image_catalog_ids, counts = np.unique(image_catalog_ids, return_counts = True)
    no_unique_images = len(unique_image_catalog_ids)
    imageList = range(0, no_unique_images)
    # sort feature list by catalog id    
    featListSorted = np.argsort(image_catalog_ids)
    # counts_cumsum[i]:counts_cumsum[i+1] are the feature indices corresponding to image i
    counts_cumsum = np.hstack([0, np.cumsum(counts)])
    
    for IID in imageList:

        # get image path
        image_catalog_id = unique_image_catalog_ids[IID]
        rasterFilePath = os.path.join(images_dir, image_catalog_id, imagery_type) 
        
        # pick first entry in path --- there should be one image file in there anyway
        rasterFile = os.listdir(rasterFilePath)[0]
        # get full path
        rasterFile = os.path.join(rasterFilePath, rasterFile)

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

        featListThisImage = featListSorted[counts_cumsum[IID]:counts_cumsum[IID+1]]

        for FID in featListThisImage:
        
            # this feature
            feat = lyr.GetFeature(FID)
             
            coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)

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
                sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")
        
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
            target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
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
            dataraster = raster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
        
            datamask = target_ds.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
        
            # replicate mask for each band
            datamask = np.dstack([datamask for i in range(nbands)])
            datamask = datamask.transpose(2,0,1)
            
            # Mask zone of raster
            zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
        
            yield ( feat, zoneraster )


def computeFeatureVectors(data):
    # This should be a lot more sophisticated!
    # Consider data normalization ( depending on classifier used )
    # Consider generating a feature vector per pixel or histograms for example
    # Consider computing additional features, e.g. band ratios etc.
    yield [ np.mean(data), np.std(data), np.var(data) ]


def train_model(polygonFile, imagery_type, classifierFile):
    """
    Compute feature vector for each polygon in polygonFile, 
    train random forest classifier and output classifier params in pickle file.
    imagery_type is the type of imagery used by the classifier    
    """          
    features = []
    labels = []
    for (feat, data) in extract_aoi_data(polygonFile, imagery_type):        
        label = feat.GetFieldAsString('class_name')
        for featureVector in computeFeatureVectors(data):
            features.append(featureVector)
            labels.append(label)
            print label, featureVector
            
    # train classifier
    X = np.array( features )
    y = np.array( labels )
    # Using a simple random forest with default parameters for this demonstration
    classifier = RandomForestClassifier()
    # train
    classifier.fit( X, y )
    # store model
    with open(classifierFile,"w") as fh:
        pickle.dump( classifier, fh )


def tfRasterToProj(x,y, geoTransform):
    """Converts coordinates from raster pixel to projected coordinate system"""    
    dfGeoX = geoTransform[0] + geoTransform[1] * x + geoTransform[2] * y;
    dfGeoY = geoTransform[3] + geoTransform[4] * x + geoTransform[5] * y;    
    return dfGeoX, dfGeoY


def sliding_window_detector(rasterFile, classifier, winX, winY, stepX, stepY):
    # Open data
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
                        
            features.append({"Class":labels[0], "geom": poly})
            
            x0 += stepX
        y0 += stepY
    
    return features


def deploy_model(rasterFile, modelFile, outputfile, winX, winY, stepX, stepY):
    with open(modelFile,"r") as fh:
        classifier = pickle.load(fh)
    results = sliding_window_detector(rasterFile, classifier, winX, winY, stepX, stepY)
    if os.path.exists(outputfile):
        os.remove(outputfile)
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
    

def main(job):
    """
    Runs the vanilla classifier pipeline.
    job is a dictionary. For the format of job.json look in the README of this repository.
    """    
    
    # Get filenames      
    train_file = job["train_file"]
    deploy_file = job["deploy_file"]
    output_file = job["output_file"]
    imagery_type = job["machine"]["params"]["imagery_type"]
    tile_size = job["machine"]["params"]["tile_size"]

    print "Train..."
    train_model(train_file, imagery_type, 'classifier.pickle')
    
    print "Deploy..."
    images_dir = os.getenv("IMAGES_DIR")
    image_catalog_id = 'fakeid1'
    imagery_type = 'pan'
    imageFile = os.path.join(images_dir, image_catalog_id, imagery_type)
    deploy_model(imageFile, 'classifier.pickle', 'output.json', tile_size, tile_size, tile_size, tile_size)
    
    print "Done"
   