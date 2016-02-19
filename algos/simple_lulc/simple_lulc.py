'''
Simple LULC.

This code is meant to illustrate the general workflow of a classification engine.

Please keep in mind that writing an actually well performing classification engine is beyond the scope of this demo.
Things are kept very very simple for illustrative purposes.

@authors:    Carsten Tusk, Kostas Stamatiou
@copyright:  2016 DigitalGlobe Inc. All rights reserved.
@contact:    ctusk@digitalglobe.com, kostas.stamatiou@digitalglobe.com
'''

import sys
import os
import pickle
import gdal, ogr, osr
import numpy as np

from sklearn.ensemble import RandomForestClassifier 
from reportlab.lib.testutils import outputfile

__version__ = 0.1
__date__ = '2016-02-19'
__updated__ = '2016-02-19'


def extract_aoi_data(polygonFile, rasterFile):
    """ Generator. Extracts pixels for each polygon in polygonFile from rasterFile and returns them as masked numpy arrays"""
    # Open data
    raster = gdal.Open(rasterFile)
    nbands = raster.RasterCount
    
    shp = ogr.Open(polygonFile)
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
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    featList = range(lyr.GetFeatureCount())

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


def computeFeatureVectors(data):
    # This should be a lot more sophisticated!
    # Consider data normalization ( depending on classifier used )
    # Consider generating a feature vector per pixel or histograms for example
    # Consider computing additional features, e.g. band ratios etc.
    yield [ np.average(data),np.mean(data),np.std(data),np.var(data) ]


def train_model(polygonFile,rasterFile, outputfile):
    # compute feature vectors for each polygon
    features = []
    labels = []
    for (feat,data) in extract_aoi_data(polygonFile, rasterFile):        
        label = feat.GetFieldAsString('rclass')
        for featureVector in computeFeatureVectors(data):
            features.append(featureVector)
            labels.append( label )
            print label, featureVector
            
    # train classifier
    X = np.array( features )
    y = np.array( labels )
    # Using a simple random forest with default parameters for this demonstration
    classifier = RandomForestClassifier()
    # train
    classifier.fit( X, y )
    # store model
    with open(outputfile,"w") as fh:
        pickle.dump( classifier, fh )

    
def apply_model(rasterFile, modelfile, outputfile):
    with open(modelfile,"r") as fh:
        classifier = pickle.load(fh)
    results = sliding_window_detector(rasterFile,classifier,50,50,50,50)
    if os.path.exists(outputfile):
        os.remove(outputfile)
    writeResults(results, outputfile)

  

def main(job_json):


