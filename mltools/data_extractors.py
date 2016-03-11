# Contains functions for extracting pixels and metadata from georeferenced imagery.

from osgeo import gdal, ogr, osr
import numpy as np


def extract_data(polygon_file, geom_sr = None):
    """Extracts pixels for each polygon in polygon_file.
       The image reference for each polygon is found in the image_name
       property of the polygon_file.
       NOTE: this function only works if all polygons are from the same image!

       Args:
           polygon_file (str): Filename. Collection of geometries in 
                               geojson or shp format.
           geom_sr (osr object): Geometry spatial reference system (srs). 
                                 If None, defaults to the polygon_file srs. 
       
       Yields:
           Geometry, pixels as masked numpy array and class name if it exists 
           (otherwise None).
    """

    # Get polygon data    
    shp = ogr.Open(polygon_file)
    lyr = shp.GetLayer()
    no_features = lyr.GetFeatureCount() 

    # Find raster identity from image_name of first polygon
    feat = lyr.GetFeature(0)
    raster_file = feat.GetFieldAsString('image_name')
    # check if raster_file has .tif extension; if not, append
    if raster_file[-4:] != '.tif':
        raster_file += '.tif'

    # Get raster info
    raster = gdal.Open(raster_file)
    nbands = raster.RasterCount
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

    for fid in xrange(no_features):

        feat = lyr.GetFeature(fid)
    
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
    
        data_mask = target_ds.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
    
        # replicate mask for each band
        data_mask = np.dstack([data_mask for i in range(nbands)])
        data_mask = data_mask.transpose(2,0,1)
        
        # Mask zone of raster
        zone_raster = np.ma.masked_array(dataraster, np.logical_not(data_mask))
        
        try:
            label = feat.GetField('class_name')
        except ValueError:
            label = None    

        yield poly, zone_raster, label

