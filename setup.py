import sys
from setuptools import setup, find_packages

open_kwds = {}
if sys.version_info > (3,):
    open_kwds['encoding'] = 'utf-8'

# with open('README.md', **open_kwds) as f:
#     readme = f.read()

# long_description=readme,
      
setup(name='mltools',
      version='0.0.1',
      description='Python tools for object detection and classification on DG imagery.',
      classifiers=[],
      keywords='',
      author='Kostas Stamatiou',
      author_email='kostas.stamatiou@digitalglobe.com',
      url='https://github.com/kostasthebarbarian/mltools',
      license='MIT',
      packages=find_packages(exclude=['docs']),
      include_package_data=True,
      zip_safe=False,
      install_requires=['numpy','json','geojson','sklearn','shapely','gdal','ogr','osr','psycopg2','psycopg2.extras']
      )
