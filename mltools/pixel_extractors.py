"""
What: Contains functions for extracting pixels from georeferenced imagery.
Authors: Carsten Tusk, Kostas Stamatiou
Contact: kostas.stamatiou@digitalglobe.com
"""

import gdal, ogr, osr
import numpy as np


def extract_data(polygon_file, raster_file, geom_sr = None):
    """Extracts pixels for each polygon in polygon_file from raster_file.

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           raster_file (str): Image filename.
           geom_sr (osr object): Geometry spatial reference system (srs). 
                                 If None, defaults to the polygon_file srs. 
       
       Yields:
           Feature object, geometry, pixels as masked numpy array and feature
           class name (if  exists, otherwise None).
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
    	
    	try:
        	label = feat.GetFieldAsString('class_name')
        except:
        	label = None	

        yield (feat, poly, zoneraster, label)
