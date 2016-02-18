
## Demo Machine. Supervised classification example.

This demo is meant to illustrate the general workflow of a classification engine.

Please keep in mind that writing an actually well performing classification engine is beyond the scope of this demo.
Things are kept very very simple for illustrative purposes, even error checking is often omitted.


### Requirements

python
numpy
scipy
scikit-learn
gdal
reportlab


### Simple training run

python demo_machine.py --train features_train.json fl_small_rgb.tif

### Simple detection run

python demo_machine.py --detect fl_small_rgb.tif

### Or all in one:

python demo_machine.py --train features_train.json --detect fl_small_rgb.tif

