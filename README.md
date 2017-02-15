# mltools

[![PyPI version](https://badge.fury.io/py/mltools.svg)](https://badge.fury.io/py/mltools)

Tools for fast prototyping of object detection and classification solutions on DG imagery.
Relies heavily on popular machine learning (ML) toolkits such as scikit-learn and deep
learning toolkits such as keras. The intent is to use mltools to experiment with algorithms;
when these are mature, they can be baked into GBDX tasks and deployed at scale on [GBDX](developer.digitalglobe.com/gbdx).  

mltools also includes a collection of auxiliary tools necessary for pre- and post- ML processing.
These are:

+ data_extractors: get pixels and metadata from DigitalGlobe imagery; uses [geoio](https://github.com/digitalglobe/geoio);
+ features: functions to derive features from pixels;
+ geojson_tools: functions to manipulate geojson files.

Example code can be found in /examples. The examples can be used as a guideline to create object detection/classification
solutions which involve one or more of the following steps:

1. train the algorithm;
2. test the algorithm on the test data and compute accuracy metrics;
3. deploy the algorithm on the target data for detection or classification.

Also check out [GBDX stories](https://github.com/PlatformStories) for examples on how mltools are used within GBDX tasks.

## Installation/Usage

For Ubuntu, install conda with the following commands (choose default options at prompt):

```bash
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh
```

For OS X, install conda with the following commands (choose default options at prompt):

```bash
wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
bash Miniconda2-latest-MacOSX-x86_64.sh
```

Then run:

```bash
bash
```
so that modifications in your .bashrc take effect.

Create a conda environment:

```bash
conda create -n env jupyter scipy gdal ephem
```

Activate the environment:

```bash
source activate env
```

Upgrade pip (if required):

```bash
pip install pip --upgrade
```

Install mltools:

```bash
pip install mltools
```

Optional: you can install the current version of the master branch with:

```bash
pip install git+https://github.com/digitalglobe/mltools
```

Keep in mind that the master branch is constantly under development.

If installation fails for some of the dependencies, (try to) install them with conda:

```bash
conda install <dependency_name>
```

and then retry:

```bash
pip install mltools
```

You can copy the scripts found in /examples in your project directory or create your own.
Keep in mind that the imagery has to be in your project folder and it should have the same name as the image_name
property in the geojson.

To exit your conda virtual environment:

```bash
source deactivate
```

## gbdxtools

The examples require imagery which can be ordered and downloaded from GBDX using [gbdxtools](http://github.com/digitalglobe/gbdxtools). You can install gbdxtools within the conda environment with:

```bash
conda install cryptography
pip install gbdxtools
```

If you have any trouble with the installation of gbdxtools, refer to the readme of the gbdxtools repo.

## Development

Activate the conda environment:

```bash
source activate env
```

Clone the repo:

```bash
git clone https://github.com/digitalglobe/mltools
cd mltools
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Please follow [this python style guide](https://google.github.io/styleguide/pyguide.html). 80-90 columns is fine.

To exit your conda virtual environment:

```bash
source deactivate
```

### Create a new version

To create a new version:

```bash
bumpversion ( major | minor | patch )
git push --tags
```

Then upload to pypi.


## Comments

[Here](https://docs.google.com/drawings/d/1tKSgFMp0lLd7Abne8CdOhb1PbdJfgCz5x9XkLwDeET0/edit?usp=sharing) is a slide my initial ideas on mltools. The vision is to use the solutions created with mltools as part of a Crowd+Machine system along the lines of [this document](https://docs.google.com/document/d/1hf82I_jDNGc0NdopXxW9RkbQjLOOGkV4lU5kdM5tqlA/edit?usp=sharing).
