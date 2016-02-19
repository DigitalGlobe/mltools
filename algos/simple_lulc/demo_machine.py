#!/bin/env python
# encoding: utf-8
'''
Demo Machine. Supervised classification example.

This demo is meant to illustrate the general workflow of a classification engine.

Please keep in mind that writing an actually well performing classification engine is beyond the scope of this demo.
Things are kept very very simple for illustrative purposes, even error checking is often omitted.

@author:     Carsten Tusk
@copyright:  2016 DigitalGlobe Inc. All rights reserved.
@contact:    ctusk@digitalglobe.com
'''

import sys
import os
import pickle
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from reportlab.lib.testutils import outputfile

import gdal, ogr, osr
import numpy as np
from sklearn.ensemble import RandomForestClassifier 

__all__ = []
__version__ = 0.1
__date__ = '2016-02-06'
__updated__ = '2016-02-06'

DEBUG = 1
TESTRUN = 0
PROFILE = 0

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

def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print 'Error Number: %s' % (err_num)
    print 'Error Type: %s' % (err_class)
    print 'Error Message: %s' % (err_msg)
    
class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg
    
def main(argv=None): # IGNORE:C0111
    '''Command line options.'''
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by Carsten Tusk on %s.
  Copyright 2016 DigitalGlobe Inc. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))
    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        parser.add_argument('--train', dest="features", help="train model using supplied features in geoJSON format")
        parser.add_argument('--detect', action="count", help="apply given model to run detection. Result is written to result.json unless otherwise specified")
        parser.add_argument('--modelfile',dest="modelfile", default="model.pickle", help="Model file to write to")
        parser.add_argument('--resultfile',dest="resultfile", default="result.json", help="File to write results to")
        parser.add_argument('imagefile')
        
        # Process arguments
        args = parser.parse_args()

        verbose = args.verbose
        if ( args.features ):
            print "Training model"
            train_model(args.features,args.imagefile,args.modelfile)
        if ( args.detect ):
            print "Running detection"
            apply_model(args.imagefile, args.modelfile, args.resultfile )
        
        print "Done"
        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception, e:
        if DEBUG or TESTRUN:
            raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

if __name__ == "__main__":
    # Register gdal error handler
    gdal.PushErrorHandler(gdal_error_handler)
    
    if DEBUG:
        #sys.argv.append("-h")
        sys.argv.append("-v")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = '_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())