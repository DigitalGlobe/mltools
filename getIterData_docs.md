# GetIterData Docs  

A class for extracting testing and training chips from multiple GeoTiff strips.

## Table of Contents
- [About](#about)
- [Inputs](#mltools.data_extractors.getiterdata)
- [Methods](#methods)
- [Examples](#examples)

## About

GetIterData can be used to iteratively generate arbitrarily-sized batches of chips with associated ids and labels from a shapefile. The output can the be used to train and/or test a convolutional neural network build in [Keras](http://keras.io/). The chips and labels are formatted to be used in the [model.fit()](http://keras.io/models/sequential/#fit) function as is.

## mltools.data_extractors.getIterData

<i>class</i> mltools.data_extractors.<b>getIterData</b>( <i>shapefile, batch_size=10000, min_chip_hw=0, max_chip_hw=125, classes = ['No swimming pool', 'Swimming pool'], return_labels = True, return_id = False, mask = True, normalize = True, props = None </i> )

| Parameter | Defualt | Type | Descrpition |
|-----------|---------|------|-------------|
| shapefile | N/A     | string | Name of the shapefile from which to extract polygons |
| batch_size | 10000 | int | Number of chips to generate per call of [create_batch()](#create_batch) |
| classes | ['Swimming_pool', 'No_swimming_pool'] | list[strings] | Name of classes that chips are sorted into. Must match the polygon class_name property exactly. |
| min_chip_hw | 0 | int | Minimum size acceptable (in pixels) for a chip. |
| max_chip_hw | 125 | int | Maximum size acceptable (in pixels) for a chip |
| return_labels | True | bool | Include labels in output of create_batch() |
| return_id | False | bool | Include the geometry id for each chip produced from create_batch(). |
| mask | True | bool | Return a masked array, replacing any pixels intensities that reside outside of the polygon with zeros. |
| normalize | True | bool | Divide all chips by 255 to keep pixel intensities between 0 and 1. |
| props | None | dictionary | If the polygons in the input shapefile come from multiple GeoTiff strips, you may define the ratio of polygons from each image to be included in the output of create_batch() It will otherwise default to the proportions of each image present in the shapefile. This argument takes the following form: {image_id_1: proportion_1, image_id_2: proportion_2} |

## Methods

1. [get_proportion](#get_proportion)
2. [yield_from_img_id](#yield_from_img_id)
3. [create_batch](#create_batch)

<i><b>\__init__</b>(shapefile, batch_size=10000, min_chip_hw=0, max_chip_hw=125, classes = ['No swimming pool', 'Swimming pool'], return_labels = True, return_id = False, mask = True, normalize = True, props = None) </i>

#### get_proportion

(property_name, property)
Get the proportion of polygons in the input shapefile that have a given attribute.

| Input | Type | Description |
|-------|------|-------------|
| property_name | string | Name of the polygon property to look through (ex: 'image_id', 'class_name') |
| property | string | The specific property to determine the proportion of in shapefile (ex: '1040010014800C00', 'Swimming pool') |
| <b> Output </b> | <b> Type </b> | <b> Description</b> |
| proportion | float | Proportion of polygons where property_name = property |

#### yield_from_img_id
(img_id, batch)
Generator that yields batches of chips from a specific image id.

| Input | Type | Description |
|-------|------|-------------|
| img_id | string | ID of the image from which to generate chips |
| batch | int | Number of chips to generate per iteration. |
| <b> Output </b> | <b> Type </b> | <b> Description </b> |
| chip generator | generator | Returns a generator object that yields batches of chips from the given image id. |

#### create_batch
Return a batch of chips from all images

| Input | Type | Description |
|-------|------|-------------|
| <i> None </i> | <i> N/A </i> | <i> N/A </i>|
| <b> Output </b> | <b> Type </b> | <b> Description</b> |
| chips | numpy array | Batch of chips |
| ids | list | Polygon ids corresponding to each chip (only if return_ids is True |
| labels | list | Polygon labels corresponding to each chip (only if return_labels is True). The output will be formatted in a tuple as follows: (chips, ids, labels) |  

## Examples

Here we will walk through an example of extracting batches of training data from a shapefile with polygons belonging to multiple images. Before starting, we must make sure that our working directory has the following:
- shapefile: the shapefile containing all of the polygons to extract
- images: each image that we will be extracting polygons from. Images should be named after their image ids and '.tif' as an extension.

Here are the contents of our working directory for this example:

    /
    └── working_directory
        ├── combined.geojson
        ├── 1040010014800C00.tif
        ├── 1040010014BF5100.tif
        ├── 1040010015050C00.tif
        └── 1040010014BCA700.tif

The combined.geojson file is a shapefile with polygons from four different images. Each polygon is categorized under the class_name property as either 'Swimming pool' or 'No swimming pool.'

1. First we create an instance of the getIterData class with our chip parameters. Open ipython and execute the following command:

        $ data_gen = getIterData('combined.geojson', batch_size=1000, min_chip_hw=20)

2. To generate a batch of training data, we now call create_batch on the instance:

        $ chips, labels = data_gen.create_batch()

We now have 1000 chips from the shapefile and their associated labels. To generate a new batch of training data simply repeat step 2.

Now let's play around with the other methods. Say we want to know the proportion of polygons in the shapefile that contain swimming pools. We can determine this using the [get_proportion](#get_proportion) method as follows:

        $ data_gen.get_proportion(property_name = 'class_name', property = 'Swimming pool')
        >> 0.05476

Cool. Now say you want 500 polygons from just one image. You can do so with the yield_from_img_id method as follows:

1. First we create the generator:

        $ single_gen = data_gen.yield_from_img_id(img_id = '1040010014BCA700', batch = 500)

2. Now we can generate a batch of chips from single_gen:

        $ chips, labels = single_gen.next()

Simply repeat step 2 to generate the next batch.
