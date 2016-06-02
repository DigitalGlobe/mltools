# Polygon Classification using Multispectral Imagery and Random Forests

## Table of Contents

1. [About](#about)
2. [Getting the imagery](#getting-the-imagery)
3. [Training and testing the classifier](#training-and-testing-the-classifier)
4. [Additional information](#additional-information)


## About

In this example, we implement a classifier which can classify polygons of arbitrary geometry as
'Contains swimming pool' or 'Does not contain swimming pool'. Classification is performed by 
computing a feature vector using WorldView-2, 8-band imagery and feeding the feature vector to a Random Forest
Classifier. The classifier is trained using a set of polygons which have been classified with high confidence
by the Tomnod crowd.

The example walks the user through the following steps:

+ Ordering the multispectral imagery from GBDX.

+ Training the classifier on a training data set and testing the classifier on a test data set.

This work was performed as part of the PSMA project whose goal is to provide a number of information layers on millions
of property parcels across urban centers in Australia.


## Getting the imagery

We will use gbdxtools (http://github.com/DigitalGlobe/gbdxtools) to order the image with catalog id 1040010014800C00
which constitutes our Area Of Interest. We will then run a workflow to produce an atmospherically 
compensated (acomped) multi-spectral image which will be used to train the classifier,
and a workflow to produce a pan-sharpened image which will be used in order to visualize our results.
Detailed information on gbdxtools can be found at http://gbdxtools.readthedocs.io/.

First, we need to activate the conda environment we created to install mltools:

        source activate env

Then install gbdxtools:

        pip install gbdxtools

Open ipython and create a gbdx interface:

        >> from gbdxtools import Interface
        >> gbdx = Interface()

Then order the image:

        >> order_id = gbdx.ordering.order('1040010014800C00')

The order might take a while. You can check on the order status as follows:

        >> result = gbdx.ordering.status(order_id)
        >> print result
        >> [{u'acquisition_id': u'1040010014800C00', u'location': u's3://receiving-dgcs-tdgplatform-com/055317677010_01_003', u'state': u'delivered'}]

The key 'location' specifies the location of the ordered image on S3. We store this because we will need it:

        >> data = result[0]['location']

We now execute the following steps in order to produce the acomped multi-spectral image and the pansharpened image.

        >> aoptask1 = gbdx.Task("AOP_Strip_Processor", data=data, enable_acomp=True)
        >> workflow1 = gbdx.Workflow([aoptask1])
        >> workflow1.savedata(aoptask1.outputs.data, location='kostas/pools/multispectral')
        >> workflow1.execute()
        >> u'4346825990110459472'
        >> aoptask2 = gbdx.Task("AOP_Strip_Processor", data=data, enable_acomp=True, enable_pansharpen=True)
        >> workflow2 = gbdx.Workflow([aoptask2])
        >> workflow2.savedata(aoptask2.outputs.data, location='kostas/pools/pansharpened')
        >> workflow2.execute()
        >> u'4346826806501479525'

The workflows might take a while to complete. We can check on their status as follows:

        >> workflow1.status
        >> {u'event': u'started', u'state': u'running'}
        >> workflow2.status
        >> {u'event': u'submitted', u'state': u'pending'}

When the workflows are done, their state will be 'complete'. This means that we can download the corresponding images locally.

        >> mkdir multispectral 
        >> gbdx.s3.download('kostas/pools/multispectral', './multispectral')
        >> mkdir pansharpened
        >> gbdx.s3.download('kostas/pools/pansharpened', './pansharpened')

These commands will download a number of files which include shapefiles and imagery metadata. 
We are only interested in the tif files. Exit ipython, go into each directory and rename the tif file to '1040010014800C00.tif'.

        > cd multispectral
        > mv 055078617010_01_assembly.tif 1040010014800C00.tif
        > cd ..
        > cd pansharpened
        > mv 055078617010_01_assembly.tif 1040010014800C00.tif
        > cd ..

You can delete the rest of the files as they will not be of any use to you in this example. 

You will have noticed that the image files are huge. You can compress them for visualization purposes 
using the following gdal command from the command line:

        > cd pansharpened
        > gdal_translate -outsize 20% 20% 1040010014800C00.tif 1040010014800C00_downsampled.tif

You can open the file in QGIS (or a regular image viewer) in order to view it.
Here is a screenshot.

<img src='images/adelaide_pansharpened.png' scale=1>   
<sub> Worldview-2 pansharpened image of Adelaide region, Australia.</sub>


## Training and testing the classifier

The training data set is found in train.geojson. (Detailed information on the geojson format can be found here: http://geojson.org/.) 
This is a collection of 1000 features (i.e., polygons, NOT to be confused with the feature vector that will be computed for each polygon by the algorithm), each consisting of a "geometry"
which includes the polygon coordinates, and a list of properties, i.e., "feature_id" (the polygon id), "image_id" (the image
catalog id on GBDX, which is 1040010014800C00 in this case) and "class_name" ("Swimming pool", "No swimming pool").
There are 500 polygons labeled "Swimming pool" and 500 polygons labeled "No swimming pool".  

The test data set is found in test.geojson. This a collection of 5000 features with the same format as in train.geojson. There are 500 features labeled "Swimming pool" and 4500 features labeled "No swimming pool". We select an imbalanced test data set because we have prior information that most properties in Adelaide do not contain swimming pools. (However, we will train the classifier on a balanced training set.)

We can visualize train.geojson and test.geojson by directly opening these files on github. If we want to visualize them overlayed on the imagery, we need to use QGIS. Here is a screenshot, with green/red polygons indicating presence/absence of a pool, respectively.

<img src='images/adelaide_dataset_snapshot.png' scale=1>   
<sub> Green/red polygons indicate presence/absence of a pools, respectively.</sub>

