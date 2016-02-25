'''
GDAL image wrapper.

Author: Carsten Tusk
Created: 02/15/2016
Contact: ctusk@digitalglobe.com
'''

import gdal, osr
from gdalconst import *
import numpy as np
import cv2 

class GDALImage:

    def __init__(self, imagefile, 
                       tilewidth=256, 
                       tileheight=256, 
                       strideX=None, 
                       strideY=None, 
                       padWithZeros=False):
    
        self.imagefile = imagefile
        self.tilewidth = tilewidth
        self.tileheight = tileheight
        # Open dataset
        self.dataset = gdal.Open(self.imagefile)
        self.nbands = self.dataset.RasterCount
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.geoTransform = self.dataset.GetGeoTransform()
        self.projRef = self.dataset.GetProjectionRef()
        self.datatype = self.dataset.GetRasterBand(1).DataType
        self.isByteImage = ( self.datatype == GDT_Byte )
        self.padWithZeros = padWithZeros

        self.strideX = strideX        
        if strideX == None:
            self.strideX = self.tilewidth

        self.strideY = strideY        
        if strideY == None:
            self.strideY = self.tileheight
                        
        # Set up projections
        self.spr = osr.SpatialReference( self.projRef )
        self.geospr = self.spr.CloneGeogCS()
                
        self.coordTfProjToGeo = osr.CoordinateTransformation( self.spr, 
                                                              self.geospr )
        self.coordTfGeoToProj = osr.CoordinateTransformation( self.geospr, 
                                                              self.spr )

    def __str__(self):
        return "%d,%d,%d" % ( self.width, self.height, self.nbands)

    def nextTile(self):
        y0 = 0
        while y0 < self.height:
            x0 = 0
            y1 = min ( y0+self.tileheight, self.height )            
            while x0 < self.width:
                x1 = min ( x0+self.tilewidth, self.width)                
                yield x0, y0, x1, y1
                x0 = x0 + self.strideX
            y0 = y0 + self.strideY

    def nextDataTile(self):
        for x0, y0, x1, y1 in self.nextTile():
            yield self.readTile(x0, y0, x1, y1), x0, y0

    def readTile(self, x0, y0, x1, y1):
        data = self.dataset.ReadAsArray(x0, y0, x1-x0, y1-y0)
        if len(data.shape) == 2:
            data = np.reshape(data, (data.shape[0], data.shape[1], 1)).transpose(2,0,1)

        if self.padWithZeros:
            if ( data.shape[1] < self.tileheight or data.shape[2] < self.tilewidth ):
                tile = np.zeros( ( data.shape[0], self.tileheight, self.tilewidth), dtype=data.dtype )
                tile[:,0:data.shape[1],0:data.shape[2]] = data[:]
                data = tile
        return data
                    
    def tfRasterToProj(self, x,y):
        dfGeoX = self.geoTransform[0] + self.geoTransform[1] * x + self.geoTransform[2] * y;
        dfGeoY = self.geoTransform[3] + self.geoTransform[4] * x + self.geoTransform[5] * y;
        return dfGeoX, dfGeoY
    
    def tfProjToRaster(self, projX, projY):
        x = ( self.geoTransform[5] * ( projX - self.geoTransform[0] ) - self.geoTransform[2] * ( projY - self.geoTransform[3] ) ) / ( self.geoTransform[5] * self.geoTransform[1] + self.geoTransform[4] * self.geoTransform[2] )
        y = (projY -  self.geoTransform[3] - x*self.geoTransform[4] ) / self.geoTransform[5]
        return x,y
    
    def tfProjToGeo(self, projx, projy):
        return self.coordTfProjToGeo.TransformPoint(projx, projy)
    
    def tfGeoToProj(self, latitude, longitude):
        return self.coordTfGeoToProj.TransformPoint(latitude, longitude)
    
    def tfGeoToRaster(self, latitude, longitude):
        proj = self.tfGeoToProj(latitude, longitude)
        return self.tfProjToRaster(proj[0], proj[1])
        
    def tfRasterToGeo(self,x,y):
        proj = self.tfRasterToProj(x, y)
        return self.tfProjToGeo( proj[0], proj[1] )
    
def stretchData(data,pct=2):
    # Linear 2pct stretch 
    a,b = np.percentile(data, [pct, 100-pct])
    return 255 * ( data - a ) / (b-a+0.001)

def createRGBImage(data, imagefile, bands=None):
    if ( bands == None):
        if len(data.shape) == 2:
            bands = [0]
        elif data.shape[0] == 3:
            bands = [2,1,0]
        elif data.shape[0] == 8:
            bands = [1,2,4]
        else:
            raise Exception("Unknown band combination and bands parameter not specified.")    

    if len(data.shape) > 2:
        data = data[ bands, :, :].transpose(1,2,0)
    
    cv2.imwrite(imagefile+'.jpg',data)        
        
if __name__ == '__main__':
    img = GDALImage("v:/skynet/cn-skynet-small.tif",700,1400,strideX=600,strideY=1300,padWithZeros=True)
    print img, img.geoTransform
    print img.projRef
    
    print list( img.nextTile() )
    
    id = 0    
    for data, x0, y0 in img.nextDataTile():
        id = id + 1
        print data.shape, x0, y0
        data = stretchData(data, 2)
        createRGBImage( data, 's:/out/tile'+str(id) )
    
        
